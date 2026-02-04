import cv2
import mediapipe as mp
import numpy as np
import torch
import collections
import time
import subprocess  # 소리/전화 기능용 (팀원 코드 아이디어)
import pickle

# 기존 우리 모듈 (경로 확인 필요)
from utils_features import FeatureExtractor
from model_definition import GRUAutoencoder, SimpleGRUAE

# =========================
# CONFIG & STATE DEFINITIONS
# =========================
SEQ_LEN = 30
MODEL_PATH = "data/models/gru_ae.pth"
SCALER_PATH = "data/models/scaler.pkl"
THRESHOLD_PATH = "data/models/threshold.txt"

# FSM States (팀원 코드 차용)
NORMAL = "NORMAL"
SUSPECT = "SUSPECT"     # 의심 (경고음) - "괜찮으세요?"
EMERGENCY = "EMERGENCY" # 응급 (전화/신고) - 10초 이상 지속 시
RECOVERY = "RECOVERY"

class HybridDMS:
    def __init__(self):
        # 1. Load GRU Model (The Brain)
        self.device = torch.device('cpu')
        try:
            self.scaler = pickle.load(open(SCALER_PATH, "rb"))
            with open(THRESHOLD_PATH, "r") as f:
                self.gru_threshold = float(f.read())
            
            # 모델 구조는 학습 때와 같아야 함 (Hidden dim 등)
            self.model = SimpleGRUAE(input_dim=2, hidden_dim=16) 
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval()
            print("[System] GRU Model Loaded Successfully.")
        except Exception as e:
            print(f"[Error] 모델 로딩 실패 (학습 먼저 하세요): {e}")
            self.gru_threshold = 0.5 # Default fallback

        self.extractor = FeatureExtractor()
        
        # Buffers
        self.feature_buffer = collections.deque(maxlen=SEQ_LEN) # For GRU
        self.raw_pitch_buffer = collections.deque(maxlen=60)    # For FFT (2 seconds @ 30fps)

        # FSM Variables
        self.state = NORMAL
        self.suspect_start_time = None
        self.emergency_start_time = None
        
        self.last_audio_time = 0
        self.call_triggered = False

    def get_fft_energy(self):
        """
        FFT를 사용해 최근 고개 움직임의 '에너지'와 '주파수 패턴'을 분석
        """
        if len(self.raw_pitch_buffer) < 30:
            return 0.0, 0.0 # Not enough data

        data = np.array(self.raw_pitch_buffer)
        
        # DC 성분(평균) 제거 -> 움직임의 변화량만 보기 위해
        data_centered = data - np.mean(data)
        
        # FFT 수행
        fft_vals = np.fft.fft(data_centered)
        fft_freq = np.fft.fftfreq(len(data_centered))
        
        # Power Spectrum (에너지)
        power = np.abs(fft_vals) ** 2
        
        # 1. Total Motion Energy (전체 움직임 에너지)
        total_energy = np.sum(power)
        
        # 2. High Frequency Energy (떨림/경련 감지용, 5Hz 이상 대역)
        # 30fps 기준, 인덱스 절반이 Nyquist frequency(15Hz)
        # 대략적인 인덱스로 고주파 대역 필터링
        idx_high = int(len(power) * 0.3) 
        high_freq_energy = np.sum(power[idx_high:])
        
        return total_energy, high_freq_energy

    def update(self, image):
        # -----------------------------
        # 1. Feature Extraction (Our 3D PnP)
        # -----------------------------
        img_h, img_w, _ = image.shape
        # (메인 루프에서 미디어파이프 처리는 이미지 넘겨받기 전 수행됨을 가정하거나 여기서 수행)
        # 여기서는 편의상 외부에서 landmarks를 넘겨받는 구조 대신, 
        # FeatureExtractor가 landmarks를 받는다고 가정하고 이미지 처리 로직은 main에 둡니다.
        pass 

    def process_frame(self, ear, pitch):
        # Data Buffering
        self.raw_pitch_buffer.append(pitch)
        
        # GRU Preprocessing
        scaled = self.scaler.transform([[ear, pitch]])[0]
        self.feature_buffer.append(scaled)

        # -----------------------------
        # 2. Hybrid Logic (GRU + FFT)
        # -----------------------------
        gru_anomaly_score = 0.0
        
        # A) GRU Anomaly Score
        if len(self.feature_buffer) == SEQ_LEN:
            seq_data = np.array(self.feature_buffer)
            input_tensor = torch.FloatTensor(seq_data).unsqueeze(0)
            with torch.no_grad():
                recon = self.model(input_tensor)
            gru_anomaly_score = torch.mean((input_tensor - recon) ** 2).item()

        # B) FFT Analysis (의식 소실 검증)
        motion_energy, tremor_energy = self.get_fft_energy()
        
        # --- JUDGEMENT LOGIC ---
        # 조건 1: GRU가 "평소와 다르다"고 판단 (Threshold 초과)
        is_abnormal_pattern = gru_anomaly_score > self.gru_threshold
        
        # 조건 2: 눈을 감고 있음 (EAR < 0.22)
        is_eyes_closed = ear < 0.22
        
        # 조건 3: 움직임이 극도로 없거나(기절) OR 너무 심함(발작) - FFT 활용
        # 에너지 500 미만이면 "축 늘어짐", 10000 이상이면 "발작" (임의 값, 튜닝 필요)
        is_motionless = motion_energy < 50.0 
        is_seizure = tremor_energy > 5000.0

        # 종합 판단: "비정상 패턴" AND ("눈감음" OR "움직임이상")
        is_danger = is_abnormal_pattern and (is_eyes_closed or is_motionless or is_seizure)

        return gru_anomaly_score, motion_energy, is_danger

    def update_fsm(self, is_danger, now):
        """ 상태 천이 (Finite State Machine) """
        
        if self.state == NORMAL:
            if is_danger:
                self.state = SUSPECT
                self.suspect_start_time = now
        
        elif self.state == SUSPECT:
            if not is_danger:
                # 회복됨 -> 다시 NORMAL
                self.state = NORMAL
                self.suspect_start_time = None
            else:
                # 위험 지속 시간 체크 (예: 3초 이상 지속 시 응급)
                if (now - self.suspect_start_time) > 3.0:
                    self.state = EMERGENCY
                    self.emergency_start_time = now
                    self.call_triggered = False

        elif self.state == EMERGENCY:
            if not is_danger:
                # 회복 모드로 전환
                self.state = RECOVERY
            else:
                # 응급 조치 (전화 걸기 등)
                self.trigger_emergency_action(now)

        elif self.state == RECOVERY:
            if is_danger:
                self.state = EMERGENCY # 재발
            else:
                # 일정 시간 안정되면 Normal로
                # (간단하게 바로 Normal로 가거나 타이머 둘 수 있음)
                self.state = NORMAL

        return self.state

    def trigger_emergency_action(self, now):
        """ 실제 알림/전화 로직 (Mac/Windows 호환을 위해 print로 대체하거나 subprocess 사용) """
        if not self.call_triggered:
            # 1. 경고음 빡세게
            if (now - self.last_audio_time) > 1.0:
                print("\a") # 윈도우 비프음
                # Mac 예시: subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff"])
                self.last_audio_time = now
            
            # 2. 10초 지나면 119/지인 호출
            if (now - self.emergency_start_time) > 10.0:
                print(">>> 🚨 EMERGENCY CALL ACTIVATED! calling 119... <<<")
                self.call_triggered = True

def main():
    # Setup MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    detector = HybridDMS()
    cap = cv2.VideoCapture(0)
    
    print("=== Hybrid DMS System Started ===")
    print("Logic: GRU(Brain) + FFT(Frequency) + FSM(State)")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        current_time = time.time()
        
        ear = 0.3 # Default (Safe)
        pitch = 0.0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Feature Extraction
                ear = detector.extractor.calculate_ear(face_landmarks.landmark)
                pitch = detector.extractor.calculate_head_pose_pitch(face_landmarks.landmark, image.shape)
                
                # Mesh Visualization
                mp.solutions.drawing_utils.draw_landmarks(
                    image=image, landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(100,100,100), thickness=1))

        # --- CORE PROCESS ---
        gru_score, fft_energy, is_danger = detector.process_frame(ear, pitch)
        current_state = detector.update_fsm(is_danger, current_time)
        
        # --- VISUALIZATION ---
        
        # 상태별 색상
        color = (0, 255, 0) # Normal (Green)
        if current_state == SUSPECT: color = (0, 255, 255) # Yellow
        elif current_state == EMERGENCY: color = (0, 0, 255) # Red
        
        # 1. State Box
        cv2.rectangle(image, (0, 0), (640, 80), (0, 0, 0), -1)
        cv2.putText(image, f"STATE: {current_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # 2. Info String
        info = f"GRU:{gru_score:.4f} | FFT:{fft_energy:.1f} | Pitch:{pitch:.0f}"
        cv2.putText(image, info, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 3. Warning Message
        if current_state == EMERGENCY:
            cv2.putText(image, "EMERGENCY! UNCONSCIOUS!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        cv2.imshow("Hybrid DMS", image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
