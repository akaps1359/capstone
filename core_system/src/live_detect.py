import cv2
import mediapipe as mp
import torch
import numpy as np
import pickle
import collections
from utils_features import FeatureExtractor
from model_definition import GRUAutoencoder, SimpleGRUAE # Import Model Class

# --- CONFIG ---
SEQ_LEN = 30 # Must match training
MODEL_PATH = "data/models/gru_ae.pth"
SCALER_PATH = "data/models/scaler.pkl"
THRESHOLD_PATH = "data/models/threshold.txt"

class LiveDetector:
    def __init__(self):
        self.device = torch.device('cpu')
        
        # Load Artifacts
        self.scaler = pickle.load(open(SCALER_PATH, "rb"))
        with open(THRESHOLD_PATH, "r") as f:
            self.threshold = float(f.read())
            
        print(f"Loaded Threshold: {self.threshold:.4f}")

        # Load Model
        # Need to know hidden_dim from training script (assuming 16)
        self.model = SimpleGRUAE(input_dim=2, hidden_dim=16)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        
        # Buffer for sequence
        self.feature_buffer = collections.deque(maxlen=SEQ_LEN)
        
        # Buffer for graph (visual history)
        self.score_history = collections.deque(maxlen=100) # Show last 100 frames
        
        # Extractor
        self.extractor = FeatureExtractor()
        
    def preprocess(self, ear, pitch):
        # Scale input
        features = np.array([[ear, pitch]])
        scaled = self.scaler.transform(features)
        return scaled[0]

    def get_anomaly_score(self):
        if len(self.feature_buffer) < SEQ_LEN:
            return 0.0
            
        # Prepare input tensor
        seq_data = np.array(self.feature_buffer) # (30, 2)
        input_tensor = torch.FloatTensor(seq_data).unsqueeze(0) # (1, 30, 2)
        
        with torch.no_grad():
            reconstruction = self.model(input_tensor)
            
        # Calculate MSE
        loss = torch.mean((input_tensor - reconstruction) ** 2).item()
        return loss

    def draw_graph(self, image, current_score):
        self.score_history.append(current_score)
        
        # Graph Config
        h, w, _ = image.shape
        graph_h, graph_w = 150, 400
        x_offset = 50
        y_offset = h - 50 # Bottom margin
        
        # Background for graph
        # cv2.rectangle(image, (x_offset, y_offset - graph_h), (x_offset + graph_w, y_offset), (0, 0, 0), -1)
        
        # Draw Threshold Line
        # We need to map score range to pixels
        # Let's say max range is 2.0 * threshold (or fixed 5.0)
        max_val = max(self.threshold * 2.0, 1.0)
        if len(self.score_history) > 0:
            max_val = max(max_val, max(self.score_history))
        
        def val_to_y(val):
            ratio = val / max_val
            px = int(ratio * graph_h)
            return y_offset - px
            
        # Draw Threshold (Red Line)
        th_y = val_to_y(self.threshold)
        cv2.line(image, (x_offset, th_y), (x_offset + graph_w, th_y), (0, 0, 255), 2)
        cv2.putText(image, f"TH: {self.threshold:.2f}", (x_offset + graph_w + 5, th_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw Score Line
        if len(self.score_history) > 1:
            pts = []
            for i, score in enumerate(self.score_history):
                x = x_offset + int((i / 100.0) * graph_w)
                y = val_to_y(score)
                pts.append((x, y))
            
            # Connect points
            for i in range(len(pts) - 1):
                p1 = pts[i]
                p2 = pts[i+1]
                
                # Color based on value
                val = self.score_history[i+1]
                color = (0, 255, 0) if val < self.threshold else (0, 0, 255)
                
                cv2.line(image, p1, p2, color, 2)
                
        # Current Value Text
        curr_y = val_to_y(current_score)
        color = (0, 255, 0) if current_score < self.threshold else (0, 0, 255)
        cv2.circle(image, (x_offset + graph_w, curr_y), 5, color, -1)
        cv2.putText(image, f"Score: {current_score:.4f}", (x_offset, y_offset - graph_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    detector = LiveDetector()
    cap = cv2.VideoCapture(0)
    
    print(">>> Starting Live Anomaly Detection...")
    print(f"Threshold: {detector.threshold}")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue
        
        # Inference
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        ear = 0.0
        pitch = 0.0
        detected = False
        
        if results.multi_face_landmarks:
            detected = True
            for face_landmarks in results.multi_face_landmarks:
                # Mesh Drawing
                mp.solutions.drawing_utils.draw_landmarks(
                   image=image, landmark_list=face_landmarks,
                   connections=mp_face_mesh.FACEMESH_TESSELATION,
                   landmark_drawing_spec=None,
                   connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(128,128,128), thickness=1, circle_radius=1))
                
                # Extract
                ear = detector.extractor.calculate_ear(face_landmarks.landmark)
                pitch = detector.extractor.calculate_head_pose_pitch(face_landmarks.landmark, image.shape)
        
        # If no face, pitch/EAR are 0.0 -> likely high anomaly score eventually, 
        # but let's just feed 0,0 or last value? 
        # Using 0.0 might be 'far' from normal (0.3 EAR, -5 Pitch).
        
        # Preprocess & Buffer
        if detected:
            scaled_feat = detector.preprocess(ear, pitch)
            detector.feature_buffer.append(scaled_feat)
        else:
            # Handling loss of tracking
            # Option A: Push zeros (will likely cause spike in anomaly) -> GOOD for detection
            # Option B: Do nothing (freeze score)
            # Let's push zeros to simulate "Abnormal State"
            detector.feature_buffer.append([0.0, 0.0]) # Or [-5.0, -5.0] to be extremel?

        # Get Score
        anomaly_score = detector.get_anomaly_score()
        
        # Visualization
        detector.draw_graph(image, anomaly_score)
        
        # Alert Logic
        if anomaly_score > detector.threshold:
            cv2.putText(image, "ANOMALY DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
        cv2.imshow("Live Anomaly Detection", image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
