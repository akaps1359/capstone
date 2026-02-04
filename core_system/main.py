import cv2
import mediapipe as mp
import time
import numpy as np

# Import our custom modules
from filters import LowPassFilter, KalmanFilter1D
from features import FeatureExtractor

def main():
    # ---------------------------
    # 1. Config & Thresholds
    # ---------------------------
    PITCH_THRESHOLD = 5.0    # 이보다 높으면 고개 숙임으로 간주
    EAR_THRESHOLD = 0.22     # 이보다 낮으면 눈 감음으로 간주
    
    # 5초 이상 지속 시 경고
    PITCH_TIME_LIMIT = 5.0
    EYE_TIME_LIMIT = 5.0
    FACE_LOST_TIME_LIMIT = 1.0 # 3.0 -> 1.0 (얼굴 놓치고 1초만 지나도 푹 숙인 걸로 의심 시작)

    # ---------------------------
    # 2. Init MediaPipe
    # ---------------------------
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # ---------------------------
    # 3. Init Filters & Features
    # ---------------------------
    pitch_kf = KalmanFilter1D(process_noise=1e-3, measurement_noise=0.1)
    ear_lpf = LowPassFilter(alpha=0.6)
    extractor = FeatureExtractor()

    # ---------------------------
    # 4. State Variables
    # ---------------------------
    pitch_timer_start = None
    eye_timer_start = None
    face_lost_timer_start = None # 얼굴 사라짐 시간 측정
    
    pitch_alert = False
    eye_alert = False
    face_lost_alert = False # 고개 푹 숙여서 얼굴 안 보임

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Webcam not found.")
        return

    print("System Started based on Python 3.11 + MediaPipe.")
    print("Press 'q' to exit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        img_h, img_w, _ = image.shape
        
        # ---------------------------
        # Logic: Face Detected vs Not Detected
        # ---------------------------
        if results.multi_face_landmarks:
            # 얼굴이 보일 때 -> '얼굴 사라짐' 타이머 리셋
            face_lost_timer_start = None 
            face_lost_alert = False

            for face_landmarks in results.multi_face_landmarks:
                raw_ear = extractor.calculate_ear(face_landmarks.landmark)
                raw_pitch = extractor.calculate_head_pose_pitch(face_landmarks.landmark, image.shape)

                filtered_ear = ear_lpf.update(raw_ear)
                filtered_pitch = pitch_kf.update(raw_pitch)

                # 1. Pitch Logic
                is_pitch_bad = filtered_pitch > PITCH_THRESHOLD 
                
                if is_pitch_bad:
                    if pitch_timer_start is None:
                        pitch_timer_start = time.time()
                    else:
                        elapsed = time.time() - pitch_timer_start
                        if elapsed >= PITCH_TIME_LIMIT:
                            pitch_alert = True
                else:
                    pitch_timer_start = None
                    pitch_alert = False

                # 2. EAR Logic
                is_eyes_closed = filtered_ear < EAR_THRESHOLD
                
                if is_eyes_closed:
                    if eye_timer_start is None:
                        eye_timer_start = time.time()
                    else:
                        elapsed = time.time() - eye_timer_start
                        if elapsed >= EYE_TIME_LIMIT:
                            eye_alert = True
                else:
                    eye_timer_start = None
                    eye_alert = False

                # Visualization
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(128,128,128), thickness=1, circle_radius=1)
                )

                # Status Text
                cv2.putText(image, f"Pitch: {filtered_pitch:.1f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not is_pitch_bad else (0, 0, 255), 2)
                
                cv2.putText(image, f"EAR: {filtered_ear:.2f}", (20, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not is_eyes_closed else (0, 0, 255), 2)

        else:
            # 얼굴이 아예 안 보일 때 (정수리만 보일 때)
            # --> 이것도 '수면/의식소실' 가능성 매우 높음
            if face_lost_timer_start is None:
                face_lost_timer_start = time.time()
            else:
                elapsed = time.time() - face_lost_timer_start
                # 얼굴 안 보인 지 3초 넘어가면 경고 격상 (안 보일 정도면 심각한 상황)
                # 사용자 요청은 5초지만, 얼굴 사라짐은 더 빨리 감지해도 됨. 일단 통일성을 위해 3초로 둠.
                # 사용자 요구사항 5초 -> 여기서는 얼굴 완전 사라짐은 3초만 지나도 경고 띄우는 게 안전함 (조정 가능)
                if elapsed >= 3.0: 
                    face_lost_alert = True
            
            # 리셋 (얼굴 없으니 Pitch/EAR 측정 불가)
            pitch_timer_start = None
            eye_timer_start = None
            pitch_alert = False 
            # eye_alert는 유지할지 말지 결정 필요. 얼굴 안 보이면 눈 상태도 모르니 일단 끔.
            eye_alert = False 

            cv2.putText(image, "NO FACE DETECTED", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ---------------------------
        # Final Alert Display
        # ---------------------------
        if face_lost_alert:
             cv2.putText(image, "WARNING: HEAD DROPPED (NO FACE)!", (50, img_h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif pitch_alert:
            cv2.putText(image, "WARNING: HEAD DOWN (5s)!", (50, img_h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        elif eye_alert:
            cv2.putText(image, "WAKE UP: EYES CLOSED (5s)!", (50, img_h//2 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        cv2.imshow('FaceMesh MVP Phase 1 (Updated)', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
