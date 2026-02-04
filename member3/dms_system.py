import cv2
import mediapipe as mp
import time
import numpy as np
import os

# [해결책] 한글 경로 및 환경 변수 문제 대응
os.environ['PYTHONUTF8'] = '1'

# MediaPipe 설정
mp_face_mesh = mp.solutions.face_mesh
# refine_landmarks=False로 설정하여 추가 모델 로드 에러 방지
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 눈의 특징점 인덱스 (MediaPipe 기본 모델 기준)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def get_ear(landmarks, eye_indices):
    """
    EAR(Eye Aspect Ratio) 수식을 사용하여 눈 감음 정도 측정 
    수식: (v1 + v2) / (2.0 * h)
    """
    # 수직 거리 계산
    v1 = np.linalg.norm(np.array(landmarks[eye_indices[1]]) - np.array(landmarks[eye_indices[5]]))
    v2 = np.linalg.norm(np.array(landmarks[eye_indices[2]]) - np.array(landmarks[eye_indices[4]]))
    # 수평 거리 계산
    h = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[3]]))
    
    if h == 0: return 0 # Zero division 방지
    return (v1 + v2) / (2.0 * h)

# 설정값
EAR_THRESHOLD = 0.22      # 눈을 감았다고 판단할 임계값 [cite: 7]
EMERGENCY_TIME = 10.0     # 위급 상황 판정 시간 (10초) [cite: 7]

cap = cv2.VideoCapture(0) # 노트북 웹캠 호출 [cite: 4]
blink_start_time = None
emergency_flag = False

print("프로그램을 시작합니다. ESC를 누르면 종료됩니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        break

    # 성능 향상을 위해 이미지 쓰기 불가능 설정 후 RGB 변환
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # 다시 BGR로 변환하여 화면 표시 준비
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            # 정규화된 좌표를 픽셀 좌표로 변환
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]

            # 양쪽 눈 EAR 계산 및 평균값 도출 
            left_ear = get_ear(landmarks, LEFT_EYE)
            right_ear = get_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            # 1. 졸음 및 위급 상황 판정 로직 [cite: 1, 7]
            if avg_ear < EAR_THRESHOLD:
                if blink_start_time is None:
                    blink_start_time = time.time()
                
                elapsed_time = time.time() - blink_start_time
                
                # 10초 이상 감았을 때 위급 상황 발생 [cite: 7]
                if elapsed_time >= EMERGENCY_TIME:
                    emergency_flag = True
                    cv2.putText(image, "!!! EMERGENCY: MRM ACTIVATED !!!", (10, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                blink_start_time = None
                emergency_flag = False

            # 2. 화면 정보 표시 (디버깅용)
            color = (0, 255, 0) if avg_ear > EAR_THRESHOLD else (0, 0, 255)
            cv2.putText(image, f"EAR: {avg_ear:.2f}", (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if blink_start_time:
                cv2.putText(image, f"Closed: {time.time()-blink_start_time:.1f}s", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 3. 갓길 정차(MRM) 로직 예시 - 실제 구현 시 조향 제어 함수 호출 [cite: 19, 21]
    if emergency_flag:
        # 여기에 Canny Edge Detection 기반 차선 인식 및 제어 코드 통합 예정 [cite: 21]
        pass

    cv2.imshow('DMS High-End System', image)

    if cv2.waitKey(5) & 0xFF == 27: # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()