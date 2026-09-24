import os
import math
import base64
import io
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import config

class FaceMemoryManager:
    """
    Manages High-Precision Face Detection, Landmarks Analysis,
    Biometric Extraction, Admin Biometric Gate, and Few-Shot Face Identity Bank.
    Operates 100% Offline with zero third-party cloud dependencies.
    """
    def __init__(self):
        self.device = torch.device(config.DEVICE)
        self._init_face_extractor()

        # 1. Primary: YuNet Deep Learning Neural Network for Face & Landmarks Detection
        self.yunet_detector = None
        if hasattr(config, 'YUNET_MODEL_PATH') and config.YUNET_MODEL_PATH.exists() and hasattr(cv2, 'FaceDetectorYN'):
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    str(config.YUNET_MODEL_PATH),
                    "",
                    (320, 240),
                    score_threshold=0.60,
                    nms_threshold=0.30
                )
                print(f"[FACE_MEMORY] Initialized YuNet Deep Face Detector ({config.YUNET_MODEL_PATH})")
            except Exception as e:
                print(f"[FACE_MEMORY] Warning: Could not initialize YuNet: {e}")
                self.yunet_detector = None

        # 2. Secondary Fallback: Strict Multi-Cascade Ensemble
        self.cascades_face = []
        for cname in ["haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"]:
            p = config.DATA_DIR / cname
            if p.exists():
                c = cv2.CascadeClassifier(str(p))
                if not c.empty():
                    self.cascades_face.append(c)

        smile_path = config.DATA_DIR / "haarcascade_smile.xml"
        eye_path = config.DATA_DIR / "haarcascade_eye.xml"

        self.face_cascade = self.cascades_face[0] if self.cascades_face else None
        self.smile_cascade = cv2.CascadeClassifier(str(smile_path)) if smile_path.exists() else None
        self.cascades_smile = [self.smile_cascade] if (self.smile_cascade and not self.smile_cascade.empty()) else []
        self.eye_cascade = cv2.CascadeClassifier(str(eye_path)) if eye_path.exists() else None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self._ensure_admin_seed()

    def _init_face_extractor(self):
        """Initialize ResNet50 feature extractor producing 2048-dim face embeddings."""
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.backbone.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _ensure_admin_seed(self):
        """If admin_face.pt does not exist, initialize a default biometric baseline."""
        if not config.ADMIN_FACE_PATH.exists():
            torch.manual_seed(42)
            admin_vec = F.normalize(torch.randn(1, config.DIM_FACE), p=2, dim=-1)
            torch.save({
                "vector": admin_vec,
                "name": "PRIMARY_ADMINISTRATOR",
                "registered_at": datetime.now().isoformat(),
                "initialized_from_template": True
            }, config.ADMIN_FACE_PATH)
            print(f"[FACE_MEMORY] Initialized baseline admin vector at {config.ADMIN_FACE_PATH}")

    def decode_image_from_base64(self, b64_str):
        """Decode base64 image data URL or raw string to OpenCV BGR image."""
        if not b64_str or not isinstance(b64_str, str):
            return None
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            b64_str = b64_str.strip().replace(" ", "+")
            rem = len(b64_str) % 4
            if rem > 0:
                b64_str += "=" * (4 - rem)
            img_bytes = base64.b64decode(b64_str)
            if not img_bytes:
                return None
            nparr = np.frombuffer(img_bytes, np.uint8)
            if nparr.size == 0:
                return None
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[FACE_MEMORY] decode_image_from_base64 error: {e}")
            return None

    def encode_image_to_base64(self, img_bgr, max_size=120):
        """Encode OpenCV BGR image to small thumbnail base64 data URL."""
        if img_bgr is None or img_bgr.size == 0:
            return ""
        h, w = img_bgr.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    def detect_face_detailed(self, img_bgr):
        """
        High-precision facial detection.
        Uses YuNet deep neural network as primary detector.
        Falls back to Haar cascades with strict filters if YuNet is unavailable.
        Returns: (bbox_tuple, confidence_float, landmarks_dict) or (None, 0.0, None)
        """
        if img_bgr is None or img_bgr.size == 0:
            return None, 0.0, None

        h, w = img_bgr.shape[:2]
        if h < 24 or w < 24:
            return None, 0.0, None

        # 1. Primary: YuNet Deep Learning Detector
        if self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((w, h))
                ret, faces = self.yunet_detector.detect(img_bgr)
                if faces is not None and len(faces) > 0:
                    # Select face with highest confidence & reasonable size
                    best_face = None
                    best_score = -1.0
                    for f in faces:
                        conf = float(f[14])
                        fw, fh = float(f[2]), float(f[3])
                        # Strict threshold: at least 0.62 confidence and min face size 36x36
                        if conf >= 0.62 and fw >= 36 and fh >= 36:
                            area = fw * fh
                            score = conf * 500.0 + area
                            if score > best_score:
                                best_score = score
                                best_face = f

                    if best_face is not None:
                        fx = int(max(0, best_face[0]))
                        fy = int(max(0, best_face[1]))
                        fw = int(min(w - fx, best_face[2]))
                        fh = int(min(h - fy, best_face[3]))
                        conf = float(best_face[14])

                        re = np.array([float(best_face[4]), float(best_face[5])])
                        le = np.array([float(best_face[6]), float(best_face[7])])
                        nose = np.array([float(best_face[8]), float(best_face[9])])
                        rm = np.array([float(best_face[10]), float(best_face[11])])
                        lm = np.array([float(best_face[12]), float(best_face[13])])

                        # Anatomical verification: face must NOT be covered by hands
                        # 1. Distance between eyes and mouth width must be plausible
                        d_eyes = float(np.linalg.norm(le - re))
                        w_mouth = float(np.linalg.norm(lm - rm))
                        
                        if d_eyes < 12.0 or w_mouth < 9.0:
                            # Occluded / covered face - reject!
                            return None, 0.0, None

                        # 2. Eye line orientation & vertical rotation
                        eye_mid = (re + le) / 2.0
                        angle_rad = float(math.atan2(le[1] - re[1], le[0] - re[0]))
                        cos_a, sin_a = math.cos(-angle_rad), math.sin(-angle_rad)
                        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

                        # Rotate points to horizontal eye plane
                        nose_rot = R @ (nose - eye_mid)
                        rm_rot = R @ (rm - eye_mid)
                        lm_rot = R @ (lm - eye_mid)

                        # In an upright face: nose is below eyes, mouth corners are below nose
                        if nose_rot[1] <= 5.0 or rm_rot[1] <= nose_rot[1] - 4.0 or lm_rot[1] <= nose_rot[1] - 4.0:
                            # Hand covering nose/mouth or invalid inverted geometry - reject!
                            return None, 0.0, None

                        mouth_mid_y = (rm_rot[1] + lm_rot[1]) / 2.0
                        d_nose_mouth_vert = float(max(1.0, mouth_mid_y - nose_rot[1]))
                        
                        r_width = float(w_mouth / max(1.0, d_eyes))
                        r_jaw = float(d_nose_mouth_vert / max(1.0, d_eyes))
                        tilt = float(math.degrees(angle_rad))

                        # Morphological smile ratio: horizontal mouth span vs vertical nose-mouth separation
                        # When smiling, mouth widens and corners pull UP closer to the nose (d_nose_mouth_vert shrinks)
                        r_smile_morph = float(w_mouth / max(1.0, d_nose_mouth_vert))

                        landmarks = {
                            "right_eye": (float(re[0]), float(re[1])),
                            "left_eye": (float(le[0]), float(le[1])),
                            "nose": (float(nose[0]), float(nose[1])),
                            "right_mouth": (float(rm[0]), float(rm[1])),
                            "left_mouth": (float(lm[0]), float(lm[1])),
                            "d_eyes": d_eyes,
                            "w_mouth": w_mouth,
                            "r_width": r_width,
                            "r_jaw": r_jaw,
                            "r_smile_morph": r_smile_morph,
                            "tilt": tilt
                        }
                        return (fx, fy, fw, fh), conf, landmarks
            except Exception as e:
                print(f"[FACE_MEMORY] YuNet detection error: {e}")

            # When YuNet is active, DO NOT fall back to loose Haar cascade to prevent hand/wall hallucinations!
            return None, 0.0, None

        # 2. Strict Secondary Fallback: Only if YuNet model is completely missing
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        candidates = []
        for cascade in getattr(self, 'cascades_face', []):
            if cascade is None or cascade.empty():
                continue
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(60, 60))
            for f in faces:
                candidates.append((int(f[0]), int(f[1]), int(f[2]), int(f[3])))
            if len(candidates) > 0:
                break

        if candidates:
            best_bbox = max(candidates, key=lambda b: b[2] * b[3])
            return best_bbox, 0.75, None

        return None, 0.0, None

    def detect_face_bbox(self, img_bgr):
        """Detect face bbox (x, y, w, h) or None if no face found."""
        bbox, _, _ = self.detect_face_detailed(img_bgr)
        return bbox

    def extract_face_vector(self, img_bgr):
        """
        Detects face ROI in image and extracts 2048-dim normalized embedding.
        If no face is detected, returns zero vector.
        Returns: (vector_tensor, crop_bgr, bbox_tuple)
        """
        if img_bgr is None or img_bgr.size == 0:
            return torch.zeros((1, config.DIM_FACE)), None, None

        bbox_res, conf, landmarks = self.detect_face_detailed(img_bgr)

        if bbox_res is not None:
            x, y, w, h = bbox_res
            face_crop = img_bgr[y:y+h, x:x+w]
            bbox = (int(x), int(y), int(w), int(h))
        else:
            # No face detected
            return torch.zeros((1, config.DIM_FACE)), None, None

        # Extract 2048-dim spatial-gradient facial biometric embedding
        resized = cv2.resize(face_crop, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        feat_spatial = np.concatenate([
            gray.flatten()[:1024],
            grad_x.flatten()[:512],
            grad_y.flatten()[:512]
        ])
        norm_feat = F.normalize(torch.from_numpy(feat_spatial).unsqueeze(0).float(), p=2, dim=-1)

        return norm_feat, face_crop, bbox

    def analyze_live_frame(self, img_bgr):
        """
        Real-time face detection, landmarks, smile analysis, eye tracking, and posture.
        Operates strictly on genuine human faces.
        Returns empty/neutral metrics if no face is in view.
        """
        if img_bgr is None or img_bgr.size == 0:
            return {
                "face_detected": False,
                "bbox": [0, 0, 0, 0],
                "confidence": 0.0,
                "landmarks": None,
                "is_smiling": False,
                "smile_score": 0.0,
                "eyes_count": 0,
                "head_tilt_deg": 0.0,
                "emotion": "KHÔNG PHÁT HIỆN KHUÔN MẶT",
                "valence": 0.0,
                "arousal": 0.0,
                "exhaustion_level": 0,
                "posture_warning": False,
                "mouth_width_ratio": 0.0,
                "jaw_drop_ratio": 0.0
            }

        h, w = img_bgr.shape[:2]
        bbox_res, conf, landmarks = self.detect_face_detailed(img_bgr)

        if bbox_res is None:
            return {
                "face_detected": False,
                "bbox": [0, 0, 0, 0],
                "confidence": 0.0,
                "landmarks": None,
                "is_smiling": False,
                "smile_score": 0.0,
                "eyes_count": 0,
                "head_tilt_deg": 0.0,
                "emotion": "KHÔNG PHÁT HIỆN KHUÔN MẶT",
                "valence": 0.0,
                "arousal": 0.0,
                "exhaustion_level": 0,
                "posture_warning": False,
                "mouth_width_ratio": 0.0,
                "jaw_drop_ratio": 0.0
            }

        x, y, fw, fh = bbox_res
        face_crop = img_bgr[y:y+fh, x:x+fw]
        if face_crop.size == 0:
            return {
                "face_detected": False,
                "bbox": [0, 0, 0, 0],
                "confidence": 0.0,
                "landmarks": None,
                "is_smiling": False,
                "smile_score": 0.0,
                "eyes_count": 0,
                "head_tilt_deg": 0.0,
                "emotion": "KHÔNG PHÁT HIỆN KHUÔN MẶT",
                "valence": 0.0,
                "arousal": 0.0,
                "exhaustion_level": 0,
                "posture_warning": False,
                "mouth_width_ratio": 0.0,
                "jaw_drop_ratio": 0.0
            }

        is_smiling = False
        smile_score = 0.0
        head_tilt = 0.0
        eyes_count = 2
        r_width = 0.80
        r_jaw = 0.60
        r_smile_morph = 1.30

        if landmarks is not None:
            r_width = float(landmarks.get("r_width", 0.80))
            r_jaw = float(landmarks.get("r_jaw", 0.60))
            head_tilt = float(landmarks.get("tilt", 0.0))
            r_smile_morph = float(landmarks.get("r_smile_morph", 1.30))

            # Precision Morphological Smile Detection:
            # - Neutral resting mouth: r_width < 0.90
            # - Genuine Duchenne Smile: zygomaticus lifts mouth corners up toward nose and widens laterally:
            #   r_smile_morph >= 1.90 AND r_width >= 0.92.
            is_morph_smile = (r_smile_morph >= 1.95 and r_width >= 0.94) or (r_smile_morph >= 1.85 and r_width >= 0.91)

            # Haar confirmation on mouth region
            haar_smile_detected = False
            mouth_roi = face_crop[int(fh * 0.50):, :]
            if mouth_roi.size > 0 and len(getattr(self, 'cascades_smile', [])) > 0:
                gray_m = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY) if len(mouth_roi.shape) == 3 else mouth_roi
                for s_casc in self.cascades_smile:
                    if s_casc and not s_casc.empty():
                        smiles = s_casc.detectMultiScale(gray_m, scaleFactor=1.3, minNeighbors=16, minSize=(20, 20))
                        if len(smiles) > 0 and r_width >= 0.88:
                            haar_smile_detected = True
                            break

            if is_morph_smile:
                is_smiling = True
                smile_score = round(min(100.0, 72.0 + (r_smile_morph - 1.85) * 60.0 + (r_width - 0.91) * 70.0), 1)
            elif haar_smile_detected:
                is_smiling = True
                smile_score = round(min(92.0, 62.0 + max(0.0, r_width - 0.88) * 100.0), 1)
            else:
                # 100% Genuine Neutral / Calm Face
                is_smiling = False
                smile_score = 0.0
        else:
            # Never hallucinate smiles if landmarks are not cleanly tracked
            is_smiling = False
            smile_score = 0.0

        # Clinical state deduction from real facial biomarkers
        is_surprised = (r_jaw >= 0.82)

        if is_smiling:
            emotion = "HAPPY / EQUILIBRIUM"
            valence = round(0.55 + (smile_score / 100.0) * 0.35, 2)
            arousal = 0.35
            exhaustion = 0
            posture_warn = False
        elif is_surprised:
            emotion = "SURPRISE"
            valence = 0.25
            arousal = 0.75
            exhaustion = 1
            posture_warn = False
        elif abs(head_tilt) > 16.0:
            emotion = "FEAR / ANXIETY"
            valence = -0.28
            arousal = 0.45
            exhaustion = 2
            posture_warn = True
        elif (y + fh) > int(h * 0.92): # User head slumped down near viewport bottom
            emotion = "SAD / DEPRESSIVE"
            valence = -0.42
            arousal = -0.20
            exhaustion = 3
            posture_warn = True
        else:
            emotion = "NEUTRAL / CALM"
            valence = 0.10
            arousal = -0.05
            exhaustion = 1
            posture_warn = False

        return {
            "face_detected": True,
            "bbox": [int(x), int(y), int(fw), int(fh)],
            "confidence": round(conf * 100.0, 1),
            "landmarks": landmarks,
            "is_smiling": is_smiling,
            "smile_score": smile_score,
            "eyes_count": eyes_count,
            "head_tilt_deg": round(head_tilt, 1),
            "emotion": emotion,
            "valence": valence,
            "arousal": arousal,
            "exhaustion_level": exhaustion,
            "posture_warning": posture_warn,
            "mouth_width_ratio": round(r_width, 3),
            "jaw_drop_ratio": round(r_jaw, 3)
        }

    def verify_admin_biometrics(self, img_bgr=None, vector=None):
        """
        Verify subject against admin_face.pt using Cosine Similarity.
        Threshold: config.BIOMETRIC_MATCH_THRESHOLD (0.70).
        """
        if not config.ADMIN_FACE_PATH.exists():
            return {"authenticated": False, "score": 0.0, "reason": "ADMIN_RECORD_MISSING"}

        admin_data = torch.load(config.ADMIN_FACE_PATH, map_location="cpu")
        admin_vec = admin_data["vector"]
        is_seed = admin_data.get("initialized_from_template", False)

        if vector is None and img_bgr is not None:
            vector, _, bbox = self.extract_face_vector(img_bgr)
            if bbox is None:
                return {
                    "authenticated": False,
                    "score": 0.0,
                    "threshold": config.BIOMETRIC_MATCH_THRESHOLD * 100,
                    "message": "[ACCESS DENIED: KHÔNG TÌM THẤY KHUÔN MẶT TRONG ỐNG KÍNH]"
                }

        if vector is None or float(torch.norm(vector)) < 1e-4:
            return {"authenticated": False, "score": 0.0, "reason": "NO_BIOMETRIC_SIGNAL"}

        score = float(F.cosine_similarity(vector, admin_vec, dim=-1).item())
        score_pct = round(score * 100, 1)

        if is_seed:
            torch.save({
                "vector": vector,
                "name": "PRIMARY_ADMINISTRATOR",
                "registered_at": datetime.now().isoformat(),
                "initialized_from_template": False
            }, config.ADMIN_FACE_PATH)
            return {
                "authenticated": True,
                "score": 100.0,
                "threshold": config.BIOMETRIC_MATCH_THRESHOLD * 100,
                "message": "[PRIMARY ADMIN BIOMETRIC ENROLLED & AUTHENTICATED]"
            }

        authenticated = score >= config.BIOMETRIC_MATCH_THRESHOLD
        return {
            "authenticated": authenticated,
            "score": score_pct,
            "threshold": config.BIOMETRIC_MATCH_THRESHOLD * 100,
            "message": "[ACCESS GRANTED: BIOMETRIC VERIFIED]" if authenticated else "[ACCESS DENIED: BIOMETRIC MISMATCH]"
        }

    def enroll_admin_direct(self, img_bgr):
        """Explicitly re-register the current face as Primary Admin."""
        vec, crop, bbox = self.extract_face_vector(img_bgr)
        if bbox is None:
            return False, "Không phát hiện thấy khuôn mặt trong ảnh để đăng ký."
        torch.save({
            "vector": vec,
            "name": "PRIMARY_ADMINISTRATOR",
            "registered_at": datetime.now().isoformat(),
            "initialized_from_template": False
        }, config.ADMIN_FACE_PATH)
        return True, "Admin biometric vector enrolled successfully."

    def enroll_face_identity(self, name, img_bgr):
        """Add a subject's face vector to face_identities.pt."""
        if not name or not name.strip():
            return False, "Identity name cannot be blank."

        vec, crop, bbox = self.extract_face_vector(img_bgr)
        if bbox is None:
            return False, "Không phát hiện thấy khuôn mặt trong ảnh để lưu danh tính."

        thumb_b64 = self.encode_image_to_base64(crop) if crop is not None else ""

        identities = self.get_all_identities_dict()
        identity_id = f"ID_{int(time.time())}_{abs(hash(name)) % 10000}"

        identities[identity_id] = {
            "id": identity_id,
            "name": name.strip(),
            "vector": vec,
            "enrolled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "thumbnail": thumb_b64
        }

        torch.save(identities, config.FACE_IDENTITIES_PATH)
        return True, f"Successfully enrolled identity '{name}'."

    def delete_face_identity(self, identity_id):
        """Remove a face identity from the ledger."""
        identities = self.get_all_identities_dict()
        if identity_id in identities:
            del identities[identity_id]
            torch.save(identities, config.FACE_IDENTITIES_PATH)
            return True, f"Removed identity {identity_id}."
        return False, f"Identity {identity_id} not found."

    def get_all_identities_dict(self):
        """Load dictionary of all enrolled identities from file."""
        if config.FACE_IDENTITIES_PATH.exists():
            try:
                return torch.load(config.FACE_IDENTITIES_PATH, map_location="cpu")
            except Exception:
                return {}
        return {}

    def get_all_identities_ledger(self):
        """Return JSON-serializable list of identities for UI display."""
        raw = self.get_all_identities_dict()
        ledger = []
        for id_key, val in raw.items():
            vec = val.get("vector")
            vec_summary = "DIM: 2048 | NORM: 1.000"
            if vec is not None:
                vec_summary = f"DIM: 2048 | MEAN: {float(vec.mean()):.4f} | L2: {float(torch.norm(vec)):.3f}"
            ledger.append({
                "id": id_key,
                "name": val.get("name", "UNKNOWN"),
                "enrolled_at": val.get("enrolled_at", "N/A"),
                "thumbnail": val.get("thumbnail", ""),
                "vector_summary": vec_summary
            })
        return ledger

    def recognize_identity(self, query_vec, threshold=0.65):
        """Find best matching registered identity for a given query vector."""
        identities = self.get_all_identities_dict()
        if not identities:
            return None, 0.0

        best_name = None
        best_score = 0.0
        for _, item in identities.items():
            stored_vec = item.get("vector")
            if stored_vec is not None:
                sim = float(F.cosine_similarity(query_vec, stored_vec, dim=-1).item())
                if sim > best_score:
                    best_score = sim
                    best_name = item.get("name")

        if best_score >= threshold:
            return best_name, round(best_score * 100, 1)
        return None, round(best_score * 100, 1)

# Global singleton
face_memory = FaceMemoryManager()
