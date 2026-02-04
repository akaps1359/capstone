import cv2
import math
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # MediaPipe FaceMesh Indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def _euclidean_dist(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def calculate_ear(self, landmarks):
        """ Calculate Eye Aspect Ratio (EAR) """
        def get_ear(indices):
            p1 = landmarks[indices[0]] # Left corner
            p2 = landmarks[indices[1]] # Top 1
            p3 = landmarks[indices[2]] # Top 2
            p4 = landmarks[indices[3]] # Right corner
            p5 = landmarks[indices[4]] # Bottom 2
            p6 = landmarks[indices[5]] # Bottom 1

            vertical_1 = self._euclidean_dist(p2, p6)
            vertical_2 = self._euclidean_dist(p3, p5)
            horizontal = self._euclidean_dist(p1, p4)

            if horizontal == 0:
                return 0.0
            
            return (vertical_1 + vertical_2) / (2.0 * horizontal)

        try:
            left_ear = get_ear(self.LEFT_EYE)
            right_ear = get_ear(self.RIGHT_EYE)
            return (left_ear + right_ear) / 2.0
        except IndexError:
            return 0.0

    def calculate_head_pose_pitch(self, landmarks, image_shape):
        """ Estimate Head Pitch (Looking down is positive) """
        img_h, img_w, _ = image_shape
        face_2d = []
        
        # Key landmarks for PnP
        key_indices = [1, 152, 33, 263, 61, 291] # Nose, Chin, L-Eye, R-Eye, L-Mouth, R-Mouth
        
        # Generic 3D model
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        for idx in key_indices:
            lm = landmarks[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])

        face_2d = np.array(face_2d, dtype=np.float64)
        focal_length = 1.0 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rot_vec, trans_vec = cv2.solvePnP(model_points, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success: return 0.0

            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            # angles[0] is pitch. MediaPipe coordinates might make this negative for 'down'.
            # We want consistency. Usually looking down makes pitch x-axis rotation specific.
            # Visual verification is best.
            return angles[0] * 360 # Scale up for readability
            
        except Exception:
            return 0.0
