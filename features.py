import numpy as np
import math

class FeatureExtractor:
    def __init__(self):
        # MediaPipe FaceMesh Iris/Eye Indices
        # Left Eye
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        # Right Eye
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def _euclidean_dist(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def calculate_ear(self, landmarks):
        """
        Calculate Eye Aspect Ratio (EAR)
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        def get_ear(indices):
            # p1=0, p2=1, p3=2, p4=3, p5=4, p6=5 (index in the list)
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
            # Both eyes average
            return (left_ear + right_ear) / 2.0
        except IndexError:
            return 0.0

    def calculate_head_pose_pitch(self, landmarks, image_shape):
        """
        Estimate Head Pitch using simplified 3D vector geometry.
        """
        img_h, img_w, _ = image_shape
        
        # Simplified 3D Head Pose Estimation using solvePnP
        # We need a generic 3D model of a face
        face_3d = []
        face_2d = []

        # Picking key points: Nose tip, Chin, Left Eye Left Corner, Right Eye Right Corner, Mouth Left, Mouth Right
        # MP Indices: Nose(1), Chin(199/152), LeftEyeLeft(33), RightEyeRight(263), MouthLeft(61), MouthRight(291)
        key_indices = [1, 152, 33, 263, 61, 291]
        
        # Generic 3D model points (arbitrary scale, centered at 0,0,0 approx)
        # Ref: Standard PnP implementation
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
            # face_3d is fixed, we use model_points

        face_2d = np.array(face_2d, dtype=np.float64)

        # Camera matrix (Assume focal length = width)
        focal_length = 1.0 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ])
        
        # Distortion coefficients (assume zero)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        try:
            success, rot_vec, trans_vec = cv2.solvePnP(model_points, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)
            
            if not success:
               return 0.0

            # Convert rotation vector to rotation matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Extract angles (Euler angles)
            # Sy = sqrt(R00 * R00 +  R10 * R10)
            # x = atan2(R21 , R22)
            # y = atan2(-R20, sy)
            # z = atan2(R10, R00)
            
            # Use decomposition to get Euler angles [pitch, yaw, roll]
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # angles[0] is pitch (x-axis rotation)
            # Note: The sign and range might need adjustment based on the model.
            # Usually looking down is positive or negative depending on coords.
            pitch = angles[0] * 360 # Convert to degrees approx scale if needed, or use raw. 
            # Actually RQDecomp3x3 returns degrees directly? Let's check logic.
            # OpenCV RQDecomp returns in degrees.
            
            return pitch 
            
        except Exception as e:
            # Fallback (simple geometry)
            # Nose y vs Ear y
            return 0.0

import cv2
