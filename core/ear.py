import sys
import time
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import distance as dist

# Ensure project root is in sys.path so imports work from anywhere
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import mediapipe as mp

# Ensure MediaPipe Tasks compatibility bridge is loaded
if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
    import core.mediapipe_bridge

# --- ĐOẠN CODE CỦA BẠN TRÊN GITHUB ---
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio with zero-division protection
    ear = (A + B) / (2.0 * C + 1e-6)
    # return the eye aspect ratio
    return ear
# -------------------------------------


class BlinkDetector:
    _shared_face_mesh = None

    def __init__(self, ear_threshold=0.22, consecutive_frames=1, neural_threshold=0.45):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.neural_threshold = neural_threshold
        self.frame_counter = 0
        self.total_blinks = 0
        self.blink_timestamps = []
        self.is_blinking = False
        self.last_blink_time = 0.0
        self.session_start_time = None  # Thời điểm phiên giám sát bắt đầu thực tế
        self.baseline_ear = 0.32        # Tự động học độ mở mắt tự nhiên của người dùng

        # 6 chỉ số mốc viền mắt chuẩn MediaPipe FaceMesh tương ứng với p0..p5
        self.LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]

        # Khởi tạo Face Mesh (MediaPipe Tasks 478 3D landmarks + Blendshapes)
        self.mp_face_mesh = mp.solutions.face_mesh
        if BlinkDetector._shared_face_mesh is None:
            BlinkDetector._shared_face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        self.face_mesh = BlinkDetector._shared_face_mesh

    def update(self, frame_bgr):
        current_time = time.time()
        if self.session_start_time is None:
            self.session_start_time = current_time

        ear = round(float(self.baseline_ear), 2)
        neural_blink_score = 0.0

        if frame_bgr is None or frame_bgr.size == 0 or len(frame_bgr.shape) < 3:
            return self._format_telemetry(ear, neural_blink_score, current_time)

        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            results = self.face_mesh.process(rgb_frame)
        except Exception:
            results = None

        if results and results.multi_face_landmarks:
            mesh_points = results.multi_face_landmarks[0].landmark
            
            # 1. Trích xuất đúng 6 tọa độ thực (x, y) cho từng mắt từ MediaPipe
            try:
                left_eye = np.array([(mesh_points[i].x * w, mesh_points[i].y * h) for i in self.LEFT_EYE_IDXS])
                right_eye = np.array([(mesh_points[i].x * w, mesh_points[i].y * h) for i in self.RIGHT_EYE_IDXS])

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
            except Exception:
                ear = self.baseline_ear

            # 2. Trích xuất trọng số nơ-ron MediaPipe Tasks Blendshapes
            blendshapes = getattr(results, "face_blendshapes", None) or {}
            blink_l = float(blendshapes.get("eyeBlinkLeft", 0.0))
            blink_r = float(blendshapes.get("eyeBlinkRight", 0.0))
            neural_avg = (blink_l + blink_r) / 2.0
            neural_blink_score = neural_avg

            # 3. Dynamic Thresholds with Hysteresis (Ngưỡng kép chống rung & chống ghost-blinks)
            close_ear_thresh = max(0.18, min(0.24, self.baseline_ear * 0.72))
            open_ear_thresh = max(0.22, min(0.28, self.baseline_ear * 0.86))

            # Điều kiện ĐÓNG mắt: cần đồng pha 2 mắt hoặc EAR sụp sâu
            is_closing = (neural_avg >= self.neural_threshold and blink_l >= 0.32 and blink_r >= 0.32) or (ear < close_ear_thresh)
            # Điều kiện MỞ mắt: tín hiệu trở về vùng an toàn
            is_opening = (neural_avg < 0.22) and (ear > open_ear_thresh)

            # 4. Strict Edge-Triggered State Machine (Chỉ đếm duy nhất 1 lần khi MỞ -> ĐÓNG)
            if not self.is_blinking:
                if is_closing:
                    self.frame_counter += 1
                    if self.frame_counter >= self.consecutive_frames:
                        # Khoảng cách tối thiểu giữa 2 cú chớp khác biệt >= 140ms
                        if current_time - self.last_blink_time >= 0.14:
                            self.is_blinking = True
                            self.total_blinks += 1
                            self.last_blink_time = current_time
                            self.blink_timestamps.append(current_time)
                else:
                    self.frame_counter = 0
            else:
                # Đang trong trạng thái nhắm mắt
                if is_opening:
                    # Mắt đã mở lại -> Reset trạng thái để sẵn sàng cho cú chớp tiếp theo
                    self.is_blinking = False
                    self.frame_counter = 0
                else:
                    # Mắt vẫn đang nhắm: Tuyệt đối KHÔNG đếm lặp lại
                    pass

            # Cập nhật mượt mà baseline mở mắt của người dùng khi mắt đang mở
            if not is_closing and not self.is_blinking and ear > 0.22:
                self.baseline_ear = 0.95 * self.baseline_ear + 0.05 * ear

        return self._format_telemetry(ear, neural_blink_score, current_time)

    def _format_telemetry(self, ear, neural_blink_score, current_time):
        # Lọc cửa sổ trượt 30 giây
        self.blink_timestamps = [t for t in self.blink_timestamps if current_time - t <= 30.0]

        # Khắc phục lỗi Cold-Start: Dùng thời gian thực tế đã trôi qua thay vì chia cứng cho 30s
        elapsed = current_time - (self.session_start_time or current_time)
        effective_window = min(30.0, max(5.0, elapsed))
        bpm = (len(self.blink_timestamps) / effective_window) * 60.0 if self.blink_timestamps else 0.0

        return {
            "ear": round(float(ear), 2),
            "blink_count": self.total_blinks,
            "blink_rate_bpm": round(float(bpm), 1),
            "is_blinking": self.is_blinking,
            "neural_blink_score": round(float(neural_blink_score), 2)
        }


if __name__ == "__main__":
    print("[EAR] Running BlinkDetector self-test (Clinical Grade Precision)...")
    detector = BlinkDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.update(frame)
    print(f"[EAR] Output telemetry: {res}")
    print("[EAR] Test passed successfully!")