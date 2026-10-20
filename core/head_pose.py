import math
import cv2
import numpy as np

# Mô hình 3D chuẩn hóa đúng 5 điểm tương ứng 1:1 với 5 landmarks của YuNet theo tọa độ camera OpenCV:
# X: sang phải (mắt phải ở bên trái camera: x âm, mắt trái ở bên phải camera: x dương)
# Y: hướng xuống (mắt ở phía trên: y âm, khóe miệng ở phía dưới: y dương)
# Z: hướng vào trong (mũi gần camera nhất: z=0, mắt và miệng lùi sâu: z dương)
YUNET_3D_MODEL_POINTS = np.array([
    (-150.0, -170.0, 135.0),     # Tâm mắt phải
    (150.0, -170.0, 135.0),      # Tâm mắt trái
    (0.0, 0.0, 0.0),             # Chóp mũi
    (-150.0, 150.0, 125.0),      # Khóe miệng phải
    (150.0, 150.0, 125.0)        # Khóe miệng trái
], dtype=np.float64)


def is_rotation_matrix(R):
    Rt = np.transpose(R)
    should_be_identity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    return np.linalg.norm(I - should_be_identity) < 1e-6


def rotation_matrix_to_euler_angles(R):
    if not is_rotation_matrix(R):
        return np.array([0.0, 0.0, 0.0])

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0

    # Trả về góc Pitch (ngửa/cúi), Yaw (quay trái/phải), Roll (nghiêng) theo độ (Degrees)
    return np.rad2deg([x, y, z])


class HeadPoseEstimator:
    def __init__(self):
        self.dist_coeffs = np.zeros((4, 1))  # Giả định camera không méo

    def estimate(self, frame_size, landmarks_5pts):
        """
        frame_size: (height, width)
        landmarks_5pts: mảng numpy shape (5, 2) chứa 5 điểm của YuNet
        """
        h, w = frame_size
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        image_points = np.array(landmarks_5pts, dtype="double")

        success, rotation_vector, translation_vector = cv2.solvePnP(
            YUNET_3D_MODEL_POINTS,
            image_points,
            camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )

        if not success:
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "head_tilt": 0.0}

        # Chuyển vector quay sang ma trận quay
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)

        pitch, yaw, roll = euler_angles[0], euler_angles[1], euler_angles[2]

        return {
            "pitch": round(float(pitch), 2),
            "yaw": round(float(yaw), 2),
            "roll": round(float(roll), 2),
            "head_tilt": round(float(abs(roll)), 2)
        }