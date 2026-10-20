"""
MediaPipe Compatibility Bridge.
Ensures mp.solutions.face_mesh is available on modern MediaPipe builds (such as on Python 3.14).
Supports official MediaPipe Tasks FaceLandmarker (data/face_landmarker.task) with automatic
Windows ctypes runtime patching and high-precision ARKit/MediaPipe blendshapes (eyeBlinkLeft/Right).
"""
import sys
import types
import ctypes
from pathlib import Path
import cv2
import numpy as np

# Method 4: Ensure project root is in sys.path so imports work from any working directory
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import config
import mediapipe as mp

# Method 1 & 2: Fix MediaPipe Tasks ctypes 'free' symbol missing on Windows C runtime
if sys.platform == "win32":
    try:
        _orig_cdll = ctypes.CDLL
        def _patched_cdll(*args, **kwargs):
            lib = _orig_cdll(*args, **kwargs)
            if not hasattr(lib, "free") and hasattr(ctypes.cdll, "msvcrt") and hasattr(ctypes.cdll.msvcrt, "free"):
                lib.free = ctypes.cdll.msvcrt.free
            return lib
        ctypes.CDLL = _patched_cdll
    except Exception:
        pass


class Point:
    __slots__ = ("x", "y", "z")
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class NormalizedLandmarkList:
    def __init__(self, points):
        self.landmark = points


class FaceMeshResults:
    def __init__(self, multi_face_landmarks=None, face_blendshapes=None):
        self.multi_face_landmarks = multi_face_landmarks
        self.face_blendshapes = face_blendshapes or {}


class FaceMeshShim:
    def __init__(self, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.max_num_faces = max_num_faces
        self.task_detector = None
        self.yunet_detector = None

        # 1. Initialize MediaPipe Tasks FaceLandmarker with Blendshapes (Method 1 & 2)
        task_model_path = config.DATA_DIR / "face_landmarker.task"
        if task_model_path.exists():
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                options = vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(task_model_path)),
                    output_face_blendshapes=True,
                    num_faces=max_num_faces,
                    min_face_detection_confidence=min_detection_confidence,
                    min_face_presence_confidence=min_tracking_confidence
                )
                self.task_detector = vision.FaceLandmarker.create_from_options(options)
            except Exception as e:
                self.task_detector = None

        # 2. Secondary Fallback: YuNet Face Detector (if .task model is missing)
        if self.task_detector is None and config.YUNET_MODEL_PATH.exists():
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    str(config.YUNET_MODEL_PATH), "", (320, 320),
                    score_threshold=0.4, nms_threshold=0.3, top_k=1
                )
            except Exception:
                self.yunet_detector = None

    def process(self, rgb_image):
        if rgb_image is None or rgb_image.size == 0:
            return FaceMeshResults(None)

        # Mode A: MediaPipe Tasks FaceLandmarker (High Precision 478 points + Neural Blendshapes)
        if self.task_detector is not None:
            try:
                contiguous_rgb = np.ascontiguousarray(rgb_image)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=contiguous_rgb)
                det_res = self.task_detector.detect(mp_img)

                if det_res.face_landmarks and len(det_res.face_landmarks) > 0:
                    blendshapes_dict = {}
                    if det_res.face_blendshapes and len(det_res.face_blendshapes) > 0:
                        for cat in det_res.face_blendshapes[0]:
                            blendshapes_dict[cat.category_name] = float(cat.score)

                    return FaceMeshResults(
                        [NormalizedLandmarkList(det_res.face_landmarks[0])],
                        face_blendshapes=blendshapes_dict
                    )
                return FaceMeshResults(None)
            except Exception:
                pass

        # Mode B: YuNet Geometric Fallback
        if self.yunet_detector is not None:
            try:
                h, w = rgb_image.shape[:2]
                self.yunet_detector.setInputSize((w, h))
                bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                ret, faces = self.yunet_detector.detect(bgr)

                if faces is None or len(faces) == 0:
                    return FaceMeshResults(None)

                f = faces[0]
                rex, rey = float(f[4]), float(f[5])  # Right eye
                lex, ley = float(f[6]), float(f[7])  # Left eye

                d_eyes = max(10.0, float(np.hypot(lex - rex, ley - rey)))
                we = max(8.0, d_eyes * 0.28)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

                def get_eye_height(cx, cy):
                    x1 = max(0, int(cx - we * 0.45))
                    x2 = min(w, int(cx + we * 0.45))
                    y1 = max(0, int(cy - we * 0.35))
                    y2 = min(h, int(cy + we * 0.35))
                    roi = gray[y1:y2, x1:x2]
                    if roi.size < 16:
                        return we * 0.32
                    col_means = np.mean(roi, axis=1)
                    contrast = float(np.max(col_means) - np.min(col_means))
                    if contrast < 16.0:
                        return we * 0.14
                    return we * 0.33

                rhe = get_eye_height(rex, rey)
                lhe = get_eye_height(lex, ley)

                pts = [Point(lex / w, ley / h) for _ in range(468)]
                # RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
                pts[33] = Point((rex - we * 0.5) / w, rey / h)
                pts[160] = Point((rex - we * 0.25) / w, (rey - rhe * 0.5) / h)
                pts[158] = Point((rex + we * 0.25) / w, (rey - rhe * 0.5) / h)
                pts[133] = Point((rex + we * 0.5) / w, rey / h)
                pts[153] = Point((rex + we * 0.25) / w, (rey + rhe * 0.5) / h)
                pts[144] = Point((rex - we * 0.25) / w, (rey + rhe * 0.5) / h)

                # LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
                pts[362] = Point((lex - we * 0.5) / w, ley / h)
                pts[385] = Point((lex - we * 0.25) / w, (ley - lhe * 0.5) / h)
                pts[387] = Point((lex + we * 0.25) / w, (ley - lhe * 0.5) / h)
                pts[263] = Point((lex + we * 0.5) / w, ley / h)
                pts[373] = Point((lex + we * 0.25) / w, (ley + lhe * 0.5) / h)
                pts[380] = Point((lex - we * 0.25) / w, (ley + lhe * 0.5) / h)

                return FaceMeshResults([NormalizedLandmarkList(pts)], face_blendshapes={})
            except Exception:
                return FaceMeshResults(None)

        return FaceMeshResults(None)


if not hasattr(mp, "solutions"):
    solutions_mod = types.ModuleType("solutions")
    face_mesh_mod = types.ModuleType("face_mesh")
    face_mesh_mod.FaceMesh = FaceMeshShim
    solutions_mod.face_mesh = face_mesh_mod
    mp.solutions = solutions_mod
else:
    if not hasattr(mp.solutions, "face_mesh"):
        face_mesh_mod = types.ModuleType("face_mesh")
        face_mesh_mod.FaceMesh = FaceMeshShim
        mp.solutions.face_mesh = face_mesh_mod


if __name__ == "__main__":
    print("[MEDIAPIPE_BRIDGE] Testing standalone execution...")
    fm = mp.solutions.face_mesh.FaceMesh()
    print(f"[MEDIAPIPE_BRIDGE] Status: Task detector={fm.task_detector is not None}, YuNet detector={fm.yunet_detector is not None}")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    res = fm.process(dummy)
    print(f"[MEDIAPIPE_BRIDGE] Test process finished cleanly. multi_face_landmarks={res.multi_face_landmarks}")
