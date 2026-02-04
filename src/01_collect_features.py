import cv2
import mediapipe as mp
import pandas as pd
import time
import os
import numpy as np
from datetime import datetime
from utils_features import FeatureExtractor

def main():
    # Setup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(0)
    
    data_buffer = []
    is_recording = False
    
    print("=== Feature Collection Tool ===")
    print("Press 'r' to toggle Recording.")
    print("Press 'q' to Quit and Save.")
    
    # Ensure data directory exists
    # Running from root: python src/01_collect_features.py
    if not os.path.exists("data/features"):
        os.makedirs("data/features", exist_ok=True)

    status_text = "READY"
    status_color = (0, 255, 0) # Green

    while cap.isOpened():
        loop_start = time.time()
        success, image = cap.read()
        if not success: continue

        # MediaPipe Processing
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        image.flags.writeable = True
        
        # Display/Logic
        ear = 0.0
        pitch = 0.0
        face_detected = False
        
        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                ear = extractor.calculate_ear(face_landmarks.landmark)
                pitch = extractor.calculate_head_pose_pitch(face_landmarks.landmark, image.shape)
                
                # Draw mesh (Visual Feedback)
                mp.solutions.drawing_utils.draw_landmarks(
                    image=image, landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(128,128,128), thickness=1, circle_radius=1)
                )

        # Recording Logic
        current_ts = time.time()
        
        if is_recording:
            status_color = (0, 0, 255) # Red
            status_text = "RECORDING..."
            
            # Save data row
            # If face not detected, we must handle it. 
            # For simplicity, we save np.nan to indicate missing data, handled in preprocessing.
            data_buffer.append({
                "timestamp": current_ts,
                "ear": ear if face_detected else np.nan,
                "pitch": pitch if face_detected else np.nan,
                "label": 0 # 0 = Normal
            })
        else:
            status_color = (0, 255, 0)
            status_text = "READY (Press 'r')"

        # UI Overlay
        cv2.putText(image, f"Status: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(image, f"Collected: {len(data_buffer)} frames", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if face_detected:
            cv2.putText(image, f"EAR: {ear:.3f} | Pitch: {pitch:.1f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(image, "NO FACE DETECTED", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.imshow("Data Collector (15FPS)", image)

        # FPS Control (Target 15 FPS -> ~66ms per frame)
        process_time = time.time() - loop_start
        # Wait at least 1ms
        wait_ms = max(1, int((0.066 - process_time) * 1000))
        
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            is_recording = not is_recording
            if is_recording:
                print(">>> Recording Started")
            else:
                print(">>> Recording Loading/Paused")

    # Save on exit if data exists
    if len(data_buffer) > 0:
        df = pd.DataFrame(data_buffer)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/features/normal_{timestamp_str}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Saved {len(df)} frames to {filename}")
        print(f"Path: {os.path.abspath(filename)}")
    else:
        print("⚠ No data collected (did you press 'r'?).")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
