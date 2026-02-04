import cv2
import mediapipe as mp
import numpy as np
import pickle
import collections
import time
from utils_features import FeatureExtractor

# --- CONFIG ---
SEQ_LEN = 30
MODEL_PATH = "data/models/pyod_model.pkl"
SCALER_PATH = "data/models/pyod_scaler.pkl"

# 성능 조절 (발열/예민함 해결)
SKIP_FRAMES = 5       # 5프레임마다 한 번만 AI 추론 (부하 대폭 감소)
SMOOTHING_WIN = 10    # 최근 10번의 점수를 평균내서 판단 (순간적인 튐 방지)
THRESHOLD_MULTIPLIER = 1.8 # PyOD가 정한 기준보다 1.8배 더 심해야 경고 (둔감하게)

class LiveDetectorPyOD:
    def __init__(self):
        try:
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
                
            # 임계값 조정 (너무 예민하면 높여야 함)
            self.base_threshold = self.model.threshold_
            self.final_threshold = self.base_threshold * THRESHOLD_MULTIPLIER
            
            print(f"✅ PyOD Loaded.")
            print(f"   Original Threshold: {self.base_threshold:.2f}")
            print(f"   Adjusted Threshold: {self.final_threshold:.2f} (x{THRESHOLD_MULTIPLIER})")
            
        except FileNotFoundError:
            print("❌ 모델 파일이 없습니다!")
            exit()
            
        self.extractor = FeatureExtractor()
        self.feature_buffer = collections.deque(maxlen=SEQ_LEN)
        
        # 점수 스무딩용 버퍼
        self.score_buffer = collections.deque(maxlen=SMOOTHING_WIN)
        
        # 지속 시간 체크용
        self.danger_start_time = None
        
    def process(self, ear, pitch):
        scaled = self.scaler.transform([[ear, pitch]])[0]
        self.feature_buffer.append(scaled)
        
        if len(self.feature_buffer) < SEQ_LEN:
            return 0.0
            
        seq_flatten = np.array(self.feature_buffer).flatten().reshape(1, -1)
        raw_score = self.model.decision_function(seq_flatten)[0]
        
        # 스무딩 (평균내기)
        self.score_buffer.append(raw_score)
        avg_score = np.mean(self.score_buffer)
        
        return avg_score

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    detector = LiveDetectorPyOD()
    cap = cv2.VideoCapture(0)
    
    threshold = detector.final_threshold
    frame_count = 0
    current_score = 0.0 # 스킵 프레임 동안 유지할 점수
    
    danger_timer_start = None
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue
        
        # 프레임 건너뛰기 로직 (부하 감소)
        frame_count += 1
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        ear = 0.3
        pitch = 0.0
        face_detected = False
        
        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                ear = detector.extractor.calculate_ear(face_landmarks.landmark)
                pitch = detector.extractor.calculate_head_pose_pitch(face_landmarks.landmark, image.shape)
                
                mp.solutions.drawing_utils.draw_landmarks(
                    image=image, landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(50,50,50), thickness=1))
        
        if not face_detected:
            # 얼굴 안 보이면 직전 값 유지 or 0 (여기선 0으로 해서 이상 감지 유도)
            ear = 0.0
            
        # 추론은 5번에 1번만! (나머지는 이전 점수 그대로 사용)
        if frame_count % SKIP_FRAMES == 0:
            current_score = detector.process(ear, pitch)
            
        # --- 시각화 및 판단 ---
        color = (0, 255, 0)
        status = "NORMAL"
        
        if current_score > threshold:
            # 경고 조건: 이상 점수가 '지속'될 때만
            if danger_timer_start is None:
                danger_timer_start = time.time()
            
            elapsed = time.time() - danger_timer_start
            
            # 2초 이상 지속되면 경고 격상 (단순 1초 튐 무시)
            if elapsed > 2.0:
                color = (0, 0, 255)
                status = f"WARNING! ({elapsed:.1f}s)"
            else:
                color = (0, 255, 255) # 노란색 (주의)
                status = "Warning..."
        else:
            danger_timer_start = None

        # 게이지 바 그리기
        max_score = threshold * 3
        bar_ratio = min(current_score / max_score, 1.0)
        
        cv2.rectangle(image, (50, 50), (350, 80), (30, 30, 30), -1)
        cv2.rectangle(image, (50, 50), (50 + int(bar_ratio * 300), 80), color, -1) # width 300
        
        # 임계값 표시선
        th_ratio = threshold / max_score
        cv2.line(image, (50 + int(th_ratio * 300), 40), (50 + int(th_ratio * 300), 90), (255, 255, 255), 2)

        cv2.putText(image, f"Status: {status}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, f"Score: {current_score:.2f} / TH: {threshold:.2f}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        
        cv2.imshow("PyOD Monitor (Optimized)", image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
