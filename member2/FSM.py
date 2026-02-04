import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import subprocess  # ✅ 추가: 소리 재생 및 시스템 명령 실행용

# =========================
# 모델 경로
# =========================
MODEL_PATH = str((Path(__file__).parent / "models" / "face_landmarker.task").resolve())

# =========================
# 랜드마크 인덱스
# =========================
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUT = 33
RIGHT_EYE_OUT = 263

# =========================
# 유틸
# =========================
def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def ear(lm, idx):
    p1 = (lm[idx[0]].x, lm[idx[0]].y)
    p2 = (lm[idx[1]].x, lm[idx[1]].y)
    p3 = (lm[idx[2]].x, lm[idx[2]].y)
    p4 = (lm[idx[3]].x, lm[idx[3]].y)
    p5 = (lm[idx[4]].x, lm[idx[4]].y)
    p6 = (lm[idx[5]].x, lm[idx[5]].y)
    denom = 2.0 * _dist(p1, p4)
    if denom <= 1e-9:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / denom

def head_pose_proxy(lm):
    """
    2D proxy:
    - yaw_proxy: 코가 눈 중심 대비 좌/우 치우침(얼굴폭 정규화)
    - pitch_proxy: 코가 눈 중심 대비 위/아래 치우침(얼굴길이 정규화)
    - cx, cy: 얼굴(눈 중심) 위치(정규화 좌표)
    """
    nose = np.array([lm[NOSE_TIP].x, lm[NOSE_TIP].y])
    chin = np.array([lm[CHIN].x, lm[CHIN].y])
    le = np.array([lm[LEFT_EYE_OUT].x, lm[LEFT_EYE_OUT].y])
    re = np.array([lm[RIGHT_EYE_OUT].x, lm[RIGHT_EYE_OUT].y])

    eye_center = (le + re) / 2.0
    face_len = np.linalg.norm(chin - eye_center) + 1e-9
    face_w = np.linalg.norm(re - le) + 1e-9

    yaw_proxy = (nose[0] - eye_center[0]) / face_w
    pitch_proxy = (nose[1] - eye_center[1]) / face_len

    cx = float(eye_center[0])
    cy = float(eye_center[1])

    return float(yaw_proxy), float(pitch_proxy), cx, cy

def robust_cov(X, eps=1e-6):
    mu = X.mean(axis=0)
    C = np.cov(X.T)
    C = C + np.eye(C.shape[0]) * eps
    return mu, C

def mahalanobis(x, mu, invC):
    d = x - mu
    return float(np.sqrt(d.T @ invC @ d))

# =========================
# FSM 상태
# =========================
NORMAL = "NORMAL"
TRANSIENT = "TRANSIENT"
SUSPECT = "SUSPECT"
EMERGENCY = "EMERGENCY"
RECOVERY = "RECOVERY"

def main():
    # =========================================================
    # 0) 파라미터 (완화 세팅 적용)
    # =========================================================
    # --- 모델(동역학 score) ---
    INIT_CALIB_SEC = 30.0
    EMA_ALPHA = 0.25
    WIN_MODEL_SEC = 180.0
    UPDATE_EVERY_SEC = 1.0

    # ✅ 완화: 회복 조건/트리거
    THR_ANOM  = 10.0   # (기존 8) "평소와 다름"
    THR_LEARN = 8.0    # (기존 5) "정상 회복" 판정도 완화

    # --- rule(의미 기반) ---
    EAR_THRESH = 0.20

    # ✅ 완화: 고개 기준 완화
    PITCH_DOWN_THRESH = 0.35  # (기존 0.28) 고개 숙임 판정 덜 민감

    # ✅ 완화: 해제 히스테리시스 줄이기
    EAR_CLEAR_MARGIN = 0.01
    PITCH_CLEAR_MARGIN = 0.01

    # 위험 지속시간
    T_TRANSIENT_TO_SUSPECT = 2.0
    T_SUSPECT_TO_EMERGENCY = 6.0

    # ✅ 완화: 복귀 시간 줄이기
    T_EMER_TO_REC = 6.0
    T_REC_TO_NORM = 10.0

    FACE_LOST_RESET_SEC = 2.0

    # --- 시각화 ---
    WIN_PLOT_SEC = 20.0
    PLOT_EVERY_N = 2  # 버벅이면 3~5

    # =========================================================
    # 1) MediaPipe
    # =========================================================
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    # =========================================================
    # 2) 버퍼/상태
    # =========================================================
    prev_time = None
    prev_feat = None
    feat_ema = None

    mu = None
    invC = None
    seed_X = []
    t0 = time.time()

    normal_samples = deque()  # (timestamp, x)
    last_update_time = 0.0

    # plot 기록
    t_hist = deque()
    score_hist = deque()
    yaw_hist = deque()
    pitch_hist = deque()

    # rule 지속시간 타이머
    ear_low_start = None
    head_down_start = None
    face_lost_start = None

    # FSM
    fsm_state = NORMAL
    transient_start = None
    suspect_start = None
    emer_stable_start = None
    rec_stable_start = None

    # ✅ 추가: 경고음 및 전화 발신용 상태 변수
    last_audio_time = 0.0
    emergency_start_time = None
    call_triggered = False

    # =========================================================
    # 3) matplotlib 패널
    # =========================================================
    fig = plt.figure(figsize=(6.2, 4.2), dpi=120)
    canvas = FigureCanvas(fig)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    fig.tight_layout(pad=2.0)

    cached_plot_bgr = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = 0.0 if prev_time is None else (now - prev_time)
        prev_time = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        score = None
        ear_ema_val = None
        pitch_val = None
        yaw_val = None

        face_detected = bool(result.face_landmarks)

        if not face_detected:
            if face_lost_start is None:
                face_lost_start = now
        else:
            face_lost_start = None

        if face_detected:
            lm = result.face_landmarks[0]

            # 1) 특징 feat: [EAR, yaw, pitch, cx, cy]
            ear_raw = (ear(lm, LEFT_EYE) + ear(lm, RIGHT_EYE)) / 2.0
            yaw_p, pitch_p, cx, cy = head_pose_proxy(lm)
            feat = np.array([ear_raw, yaw_p, pitch_p, cx, cy], dtype=np.float32)

            # 2) EMA
            if feat_ema is None:
                feat_ema = feat.copy()
            else:
                feat_ema = EMA_ALPHA * feat + (1 - EMA_ALPHA) * feat_ema

            ear_ema_val = float(feat_ema[0])
            yaw_val = float(feat_ema[1])
            pitch_val = float(feat_ema[2])

            # 3) dfeat/dt
            if prev_feat is None or dt <= 1e-6:
                dfeat = np.zeros_like(feat_ema)
            else:
                dfeat = (feat_ema - prev_feat) / dt
            prev_feat = feat_ema.copy()

            x = np.concatenate([feat_ema, dfeat], axis=0)

            # seed 캘리브레이션
            if mu is None:
                seed_X.append(x)
                if (now - t0) >= INIT_CALIB_SEC and len(seed_X) > 80:
                    X = np.stack(seed_X, axis=0)
                    mu0, C0 = robust_cov(X, eps=1e-5)
                    invC0 = np.linalg.inv(C0)
                    mu, invC = mu0, invC0

                    for i, xi in enumerate(seed_X):
                        normal_samples.append((now - (len(seed_X)-i)*0.01, xi))
                    last_update_time = now

            # score + 적응 업데이트
            if mu is not None and invC is not None:
                score = mahalanobis(x, mu, invC)

                # 정상 샘플 누적(게이팅)
                if score < THR_LEARN:
                    normal_samples.append((now, x))

                while normal_samples and (now - normal_samples[0][0]) > WIN_MODEL_SEC:
                    normal_samples.popleft()

                # 모델 업데이트(이상구간에서는 freeze)
                if (now - last_update_time) >= UPDATE_EVERY_SEC:
                    last_update_time = now
                    if score < THR_ANOM and len(normal_samples) >= 120:
                        Xn = np.stack([s[1] for s in normal_samples], axis=0)
                        mu_new, C_new = robust_cov(Xn, eps=1e-5)
                        invC_new = np.linalg.inv(C_new)
                        mu, invC = mu_new, invC_new

            # rule 지속시간
            ear_low = (ear_ema_val is not None) and (ear_ema_val < EAR_THRESH)
            head_down = (pitch_val is not None) and (pitch_val > PITCH_DOWN_THRESH)

            if ear_low:
                if ear_low_start is None:
                    ear_low_start = now
            else:
                ear_low_start = None

            if head_down:
                if head_down_start is None:
                    head_down_start = now
            else:
                head_down_start = None

            ear_low_dur = 0.0 if ear_low_start is None else (now - ear_low_start)
            head_down_dur = 0.0 if head_down_start is None else (now - head_down_start)

            # FSM
            if score is None:
                fsm_state = NORMAL
                transient_start = None
                suspect_start = None
                emer_stable_start = None
                rec_stable_start = None
            else:
                deviated = (score >= THR_ANOM)
                recovered = (score < THR_LEARN)
                danger_now = (ear_low or head_down)

                # 해제용 안정 조건(히스테리시스)
                ear_clear = (ear_ema_val >= (EAR_THRESH + EAR_CLEAR_MARGIN))
                pitch_clear = (pitch_val <= (PITCH_DOWN_THRESH - PITCH_CLEAR_MARGIN))
                stable_now = recovered and (ear_clear and pitch_clear)


                if fsm_state == NORMAL:
                    if deviated:
                        fsm_state = TRANSIENT
                        transient_start = now

                elif fsm_state == TRANSIENT:
                    if recovered:
                        fsm_state = NORMAL
                        transient_start = None
                        suspect_start = None
                    else:
                        if danger_now and transient_start is not None and (now - transient_start) >= T_TRANSIENT_TO_SUSPECT:
                            fsm_state = SUSPECT
                            suspect_start = now

                elif fsm_state == SUSPECT:
                    if recovered and (not danger_now):
                        fsm_state = NORMAL
                        transient_start = None
                        suspect_start = None
                    else:
                        if max(ear_low_dur, head_down_dur) >= T_SUSPECT_TO_EMERGENCY:
                            fsm_state = EMERGENCY
                            emer_stable_start = None
                            rec_stable_start = None

                elif fsm_state == EMERGENCY:
                    # 안정 조건이 6초 연속이면 RECOVERY로
                    if stable_now:
                        if emer_stable_start is None:
                            emer_stable_start = now
                        elif (now - emer_stable_start) >= T_EMER_TO_REC:
                            fsm_state = RECOVERY
                            rec_stable_start = now
                    else:
                        emer_stable_start = None

                elif fsm_state == RECOVERY:
                    # 재발하면 즉시 EMERGENCY
                    if danger_now or deviated:
                        fsm_state = EMERGENCY
                        emer_stable_start = None
                        rec_stable_start = None
                    else:
                        # 안정이 10초 연속이면 NORMAL
                        if stable_now:
                            if rec_stable_start is None:
                                rec_stable_start = now
                            elif (now - rec_stable_start) >= T_REC_TO_NORM:
                                fsm_state = NORMAL
                                transient_start = None
                                suspect_start = None
                                emer_stable_start = None
                                rec_stable_start = None
                        else:
                            rec_stable_start = None

            # 디버그 점
            h, w = frame.shape[:2]
            for idx in LEFT_EYE + RIGHT_EYE:
                xpx = int(lm[idx].x * w)
                ypx = int(lm[idx].y * h)
                cv2.circle(frame, (xpx, ypx), 2, (0, 255, 0), -1)

            # plot 기록
            t_hist.append(now)
            score_hist.append(float(score) if score is not None else np.nan)
            yaw_hist.append(float(yaw_val))
            pitch_hist.append(float(pitch_val))

            while t_hist and (now - t_hist[0]) > WIN_PLOT_SEC:
                t_hist.popleft()
                score_hist.popleft()
                yaw_hist.popleft()
                pitch_hist.popleft()

        # 얼굴 미검출 오래가면 RECOVERY 해제 방지(선택)
        if face_lost_start is not None:
            lost_dur = now - face_lost_start
            if lost_dur >= FACE_LOST_RESET_SEC:
                if fsm_state == RECOVERY:
                    fsm_state = EMERGENCY
                    emer_stable_start = None
                    rec_stable_start = None

        # 얼굴 미검출 오래가면 RECOVERY 해제 방지(선택)
        if face_lost_start is not None:
            lost_dur = now - face_lost_start
            if lost_dur >= FACE_LOST_RESET_SEC:
                if fsm_state == RECOVERY:
                    fsm_state = EMERGENCY
                    emer_stable_start = None
                    rec_stable_start = None

        # =========================================================
        # ✅ 추가 기능: 경고음(SUSPECT) & 전화(EMERGENCY)
        # =========================================================
        # 1) SUSPECT 단계: 경고음 (20% 볼륨, 3초 간격)
        if fsm_state == SUSPECT:
            if (now - last_audio_time) > 3.0:
                # -v 0.2 옵션으로 볼륨 20% 설정 (macOS afplay)
                subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff", "-v", "0.2"])
                last_audio_time = now

        # 2) EMERGENCY 단계: 10초 지속 시 전화 걸기
        if fsm_state == EMERGENCY:
            if emergency_start_time is None:
                emergency_start_time = now
            
            # 10초 지났고, 아직 전화를 안 걸었으면 -> 전화 걸기
            if (now - emergency_start_time) > 10.0 and not call_triggered:
                print("[WARN] EMERGENCY 10s detected! Calling user...")
                # macOS 전화 걸기 팝업 (실제 발신은 클릭 필요)
                # 전화번호는 예시로 넣어둠
                subprocess.Popen(["open", "tel://01012345678"])
                call_triggered = True
        else:
            # EMERGENCY 아니면 타이머/트리거 리셋
            emergency_start_time = None
            call_triggered = False
        # UI
        # =========================================================
        if fsm_state == EMERGENCY:
            box_color = (0, 0, 255)
        elif fsm_state == SUSPECT:
            box_color = (0, 165, 255)
        elif fsm_state == TRANSIENT:
            box_color = (255, 255, 0)
        elif fsm_state == RECOVERY:
            box_color = (255, 0, 255)
        else:
            box_color = (0, 255, 0)

        cv2.rectangle(frame, (10, 10), (920, 175), (0, 0, 0), -1)

        if mu is None:
            cv2.putText(frame, f"MODEL: SEED_CALIB {min(now - t0, INIT_CALIB_SEC):.1f}/{INIT_CALIB_SEC:.0f}s",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "Keep normal posture during seed calibration",
                        (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        else:
            cv2.putText(frame, f"MODEL: ADAPTIVE (win={WIN_MODEL_SEC:.0f}s) normal_buf={len(normal_samples)}",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.putText(frame, f"FSM: {fsm_state}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, box_color, 3)

        # ✅ 추가: SUSPECT 경고 문구
        if fsm_state == SUSPECT:
             cv2.putText(frame, "WARNING: ARE YOU OKAY?", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
        
        # ✅ 추가: EMERGENCY 전화 카운트다운 표시
        if fsm_state == EMERGENCY and emergency_start_time is not None:
             emer_dur = now - emergency_start_time
             cv2.putText(frame, f"CALLING IN {max(0, 10.0 - emer_dur):.1f}s", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
             if call_triggered:
                 cv2.putText(frame, "CALL TRIGGERED!", (50, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        s_txt = "-" if score is None else f"{score:.2f}"
        cv2.putText(frame, f"score={s_txt}  THR_ANOM={THR_ANOM:.1f}  THR_LEARN={THR_LEARN:.1f}",
                    (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if (ear_ema_val is not None) and (pitch_val is not None):
            ear_low = ear_ema_val < EAR_THRESH
            head_down = pitch_val > PITCH_DOWN_THRESH
            ear_low_dur = 0.0 if ear_low_start is None else (now - ear_low_start)
            head_down_dur = 0.0 if head_down_start is None else (now - head_down_start)

            cv2.putText(frame, f"EAR={ear_ema_val:.3f} (<{EAR_THRESH:.2f}? {ear_low}) dur={ear_low_dur:.1f}s",
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"pitch={pitch_val:.3f} (>{PITCH_DOWN_THRESH:.2f}? {head_down}) dur={head_down_dur:.1f}s",
                        (420, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if fsm_state == EMERGENCY:
                prog = 0.0 if emer_stable_start is None else (now - emer_stable_start)
                cv2.putText(frame, f"EMER->REC stable {prog:.1f}/{T_EMER_TO_REC:.0f}s",
                            (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            elif fsm_state == RECOVERY:
                prog = 0.0 if rec_stable_start is None else (now - rec_stable_start)
                cv2.putText(frame, f"REC->NORM stable {prog:.1f}/{T_REC_TO_NORM:.0f}s (re-fall -> EMERGENCY)",
                            (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

        # =========================================================
        # plot 업데이트
        # =========================================================
        frame_count += 1
        if (frame_count % PLOT_EVERY_N == 0) and len(t_hist) >= 5:
            ax1.clear()
            ax2.clear()

            t_base = t_hist[0]
            tt = np.array([t - t_base for t in t_hist], dtype=np.float32)
            ss = np.array(score_hist, dtype=np.float32)

            ax1.set_title("Anomaly score vs time (recent)")
            ax1.set_xlabel("time (s)")
            ax1.set_ylabel("score")
            ax1.plot(tt, ss)
            ax1.axhline(THR_ANOM, linestyle="--")
            ax1.axhline(THR_LEARN, linestyle=":")

            yy = np.array(yaw_hist, dtype=np.float32)
            pp = np.array(pitch_hist, dtype=np.float32)
            ax2.set_title("State space: yaw_proxy vs pitch_proxy (recent)")
            ax2.set_xlabel("yaw_proxy")
            ax2.set_ylabel("pitch_proxy")
            ax2.plot(yy, pp)

            fig.tight_layout(pad=2.0)
            canvas.draw()

            w_can, h_can = canvas.get_width_height()
            buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
            buf = buf.reshape((h_can, w_can, 4))
            plot_rgb = buf[:, :, [1, 2, 3]]
            plot_bgr = cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR)
            cached_plot_bgr = plot_bgr

        # =========================================================
        # 화면 합치기
        # =========================================================
        if cached_plot_bgr is None:
            combined = frame
        else:
            ph, pw = cached_plot_bgr.shape[:2]
            fh, fw = frame.shape[:2]
            plot_resized = cv2.resize(cached_plot_bgr, (int(pw * (fh / ph)), fh))
            combined = np.hstack([frame, plot_resized])

        cv2.imshow("DMS: Adaptive + Rule + FSM(Recovery, relaxed) | q to quit", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()

