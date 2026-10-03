import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MEMES_DIR = STATIC_DIR / "assets" / "memes"

# Ensure all critical directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
MEMES_DIR.mkdir(parents=True, exist_ok=True)

# Hardware setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Biometrics & Model Dimensions (arXiv:2412.17907 backbone specs)
DIM_FACE = 2048      # ResNet50 face embeddings
DIM_BODY = 132       # MediaPipe Pose 33 3D keypoints x 4 (x, y, z, visibility)
DIM_SPEECH = 768     # 16kHz audio spectral / acoustic feature projection
DIM_TEXT = 768       # Semantic text embedding projection
DIM_FUSED = 256      # Joint GMU embedding space

BIOMETRIC_MATCH_THRESHOLD = 0.70

EMOTION_CLASSES = [
    "HAPPY / EQUILIBRIUM",
    "SAD / DEPRESSIVE",
    "ANGRY / HIGH AROUSAL",
    "NEUTRAL / CALM",
    "SURPRISE",
    "FEAR / ANXIETY"
]

# File storage paths
ADMIN_FACE_PATH = DATA_DIR / "admin_face.pt"
FACE_IDENTITIES_PATH = DATA_DIR / "face_identities.pt"
CUSTOM_SAMPLES_PATH = DATA_DIR / "custom_samples.pt"
CUSTOM_KNOWLEDGE_PATH = DATA_DIR / "custom_knowledge.json"
CHECKPOINT_REGISTRY_PATH = CHECKPOINTS_DIR / "model_registry.json"
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_fusion_model.pt"
YUNET_MODEL_PATH = DATA_DIR / "face_detection_yunet_2023mar.onnx"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
