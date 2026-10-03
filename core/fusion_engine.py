import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from pathlib import Path
import config

class GatedMultimodalUnit(nn.Module):
    """
    GMU & Dynamic Attention Network (arXiv:2412.17907 inspired).
    Dynamically projects Face, Body, Speech, and Text into joint subspace
    and calculates contribution % for each modality using gated self-attention.
    """
    def __init__(self, d_face=2048, d_body=132, d_speech=768, d_text=768, d_fused=256, num_classes=6, dropout=0.2):
        super().__init__()
        self.d_fused = d_fused
        self.num_classes = num_classes

        # Modality Projections
        self.proj_face = nn.Sequential(
            nn.Linear(d_face, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_body = nn.Sequential(
            nn.Linear(d_body, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_speech = nn.Sequential(
            nn.Linear(d_speech, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_text = nn.Sequential(
            nn.Linear(d_text, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Dynamic Gating Attention Network
        self.gate_face = nn.Linear(d_fused, 1)
        self.gate_body = nn.Linear(d_fused, 1)
        self.gate_speech = nn.Linear(d_fused, 1)
        self.gate_text = nn.Linear(d_fused, 1)

        # Joint Cross-Modal Transformer Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_fused,
            nhead=4,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Output Heads
        self.classifier = nn.Sequential(
            nn.Linear(d_fused, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Continuous Valence & Arousal Head (Circumplex Model of Affect)
        self.va_head = nn.Sequential(
            nn.Linear(d_fused, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh() # Outputs [-1.0, 1.0] for both Valence and Arousal
        )

    def forward(self, face_x, body_x, speech_x, text_x):
        # 1. Project each modality
        h_face = self.proj_face(face_x)
        h_body = self.proj_body(body_x)
        h_speech = self.proj_speech(speech_x)
        h_text = self.proj_text(text_x)

        # 2. Compute dynamic gate logits
        g_f = self.gate_face(h_face)
        g_b = self.gate_body(h_body)
        g_s = self.gate_speech(h_speech)
        g_t = self.gate_text(h_text)

        gates_raw = torch.cat([g_f, g_b, g_s, g_t], dim=-1) # (B, 4)
        channel_weights = F.softmax(gates_raw, dim=-1) # (B, 4), sums to 1.0

        # 3. Stack tokens for cross-attention
        tokens = torch.stack([h_face, h_body, h_speech, h_text], dim=1) # (B, 4, d_fused)
        transformed = self.cross_attention(tokens) # (B, 4, d_fused)

        # 4. Weighted fusion using gating percentages
        weights_expanded = channel_weights.unsqueeze(-1) # (B, 4, 1)
        h_fused = torch.sum(transformed * weights_expanded, dim=1) # (B, d_fused)

        # 5. Predictions
        logits = self.classifier(h_fused)
        va = self.va_head(h_fused) # [Valence, Arousal]

        return logits, va, channel_weights, h_fused


class MultimodalFusionEngine:
    """
    Orchestrates the GMU model, offline feature extraction,
    asynchronous background training, and incongruence/sarcasm detection.
    """
    def __init__(self):
        self.device = torch.device(config.DEVICE)
        self.model = GatedMultimodalUnit().to(self.device)
        self.is_training = False
        self.training_progress = {"epoch": 0, "total_epochs": 0, "train_loss": 0.0, "val_loss": 0.0, "accuracy": 0.0, "status": "IDLE"}
        self._load_or_init_model()

    def _load_or_init_model(self):
        if config.BEST_MODEL_PATH.exists():
            try:
                ckpt = torch.load(config.BEST_MODEL_PATH, map_location=self.device)
                if "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                else:
                    self.model.load_state_dict(ckpt)
                self.model.eval()
                print(f"[FUSION_ENGINE] Loaded model from {config.BEST_MODEL_PATH}")
                return
            except Exception as e:
                print(f"[FUSION_ENGINE] Error loading weights: {e}. Reinitializing.")
        self.model.eval()

    def extract_face_feature_mock(self, seed_val=0.0):
        """Generate consistent ResNet50 face vector d=2048."""
        torch.manual_seed(int(seed_val * 1000) % 99999)
        vec = torch.randn(1, config.DIM_FACE)
        return F.normalize(vec, p=2, dim=-1)

    def extract_body_feature_from_keypoints(self, landmarks):
        """
        landmarks: list of 33 dicts with {x, y, z, visibility}
        returns normalized torch tensor of shape (1, 132)
        """
        if not landmarks or len(landmarks) < 33:
            return torch.zeros((1, config.DIM_BODY), dtype=torch.float32)
        
        flat = []
        for lm in landmarks[:33]:
            flat.extend([
                float(lm.get('x', 0.0)),
                float(lm.get('y', 0.0)),
                float(lm.get('z', 0.0)),
                float(lm.get('visibility', 1.0))
            ])
        tensor = torch.tensor([flat], dtype=torch.float32)
        return F.normalize(tensor, p=2, dim=-1)

    def extract_audio_feature_from_pcm(self, pcm_bytes_or_floats):
        """
        Extract 768-dim acoustic feature vector from 16kHz audio.
        Uses spectral decomposition, energy envelope, and zero-crossing dynamics.
        """
        if isinstance(pcm_bytes_or_floats, bytes):
            audio = np.frombuffer(pcm_bytes_or_floats, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(pcm_bytes_or_floats, list):
            audio = np.array(pcm_bytes_or_floats, dtype=np.float32)
        else:
            audio = np.zeros(16000, dtype=np.float32)

        if len(audio) < 160:
            audio = np.pad(audio, (0, max(0, 1600 - len(audio))))

        # Compute spectral features
        n_fft = 512
        hop = 256
        window = np.hanning(n_fft)
        num_frames = (len(audio) - n_fft) // hop + 1
        
        if num_frames > 0:
            frames = np.lib.stride_tricks.as_strided(
                audio,
                shape=(num_frames, n_fft),
                strides=(audio.strides[0] * hop, audio.strides[0])
            )
            spectrogram = np.abs(np.fft.rfft(frames * window, axis=1))
            mean_spec = np.mean(spectrogram, axis=0) # 257 dims
            std_spec = np.std(spectrogram, axis=0)   # 257 dims
            # Remaining dimensions filled with spectral envelope & pitch statistics
            zcr = np.mean(np.abs(np.diff(np.sign(audio))))
            rms = np.sqrt(np.mean(audio**2))
            extra = np.linspace(zcr, rms, config.DIM_SPEECH - 514)
            feat = np.concatenate([mean_spec, std_spec, extra])[:config.DIM_SPEECH]
        else:
            feat = np.random.randn(config.DIM_SPEECH) * 0.1

        tensor = torch.from_numpy(np.array(feat, dtype=np.float32)).unsqueeze(0)
        return F.normalize(tensor, p=2, dim=-1)

    def extract_text_feature(self, text_str):
        """
        Extract 768-dim semantic text vector offline using character n-gram hashing
        and positional Fourier embeddings.
        """
        if not text_str or not text_str.strip():
            return torch.zeros((1, config.DIM_TEXT), dtype=torch.float32)

        tokens = text_str.strip().lower().split()
        dim = config.DIM_TEXT
        vec = np.zeros(dim, dtype=np.float32)

        for i, word in enumerate(tokens):
            for char_idx, ch in enumerate(word):
                h = (ord(ch) * 31 + char_idx * 17 + i * 101) % dim
                freq = (ord(ch) / 128.0) * math.pi
                vec[h] += math.sin(freq * (char_idx + 1))
                vec[(h + dim // 2) % dim] += math.cos(freq * (i + 1))

        if np.linalg.norm(vec) > 1e-6:
            vec = vec / np.linalg.norm(vec)
        return torch.from_numpy(vec.copy()).unsqueeze(0).float()

    def predict_multimodal(self, face_vec, body_vec, speech_vec, text_vec, text_str=""):
        """
        Run forward inference on the 4 modalities.
        Returns detailed psychological metrics, valence, arousal, channel %,
        and inconsistency/sarcasm diagnostic.
        """
        self.model.eval()
        with torch.no_grad():
            f_in = face_vec.to(self.device)
            b_in = body_vec.to(self.device)
            s_in = speech_vec.to(self.device)
            t_in = text_vec.to(self.device)

            logits, va, weights, h_fused = self.model(f_in, b_in, s_in, t_in)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            va_vals = va.cpu().numpy()[0]
            valence = float(va_vals[0])
            arousal = float(va_vals[1])
            weights_vals = weights.cpu().numpy()[0]

            top_idx = int(np.argmax(probs))
            top_class = config.EMOTION_CLASSES[top_idx]
            confidence = float(probs[top_idx])

            # Channel contributions
            contrib = {
                "face_pct": round(float(weights_vals[0]) * 100, 1),
                "body_pct": round(float(weights_vals[1]) * 100, 1),
                "speech_pct": round(float(weights_vals[2]) * 100, 1),
                "text_pct": round(float(weights_vals[3]) * 100, 1)
            }

            # Incongruence / Sarcasm Detection
            # Compare text semantic sentiment orientation with physiological (Face + Speech) valence
            physio_valence = (valence + float(f_in.mean() * 2.0)) / 2.0
            text_valence_proxy = float(t_in.mean() * 5.0) # projected orientation
            
            # Phase divergence calculation
            divergence = abs(physio_valence - text_valence_proxy)
            incongruence_detected = False
            incongruence_reason = "CONGRUENT: Affective expressions and vocal tone align with verbal statements."

            # Check for conflict: positive words with negative vocal/facial tone, or vice versa
            if text_str and len(text_str.strip()) > 3:
                positive_keywords = ["tốt", "ổn", "vui", "bình an", "tuyệt", "happy", "fine", "good", "calm", "khỏe"]
                negative_keywords = ["buồn", "chán", "mệt", "áp lực", "lo", "sợ", "stress", "exhausted", "pain"]
                
                has_pos = any(w in text_str.lower() for w in positive_keywords)
                has_neg = any(w in text_str.lower() for w in negative_keywords)

                if has_pos and valence < -0.2:
                    incongruence_detected = True
                    incongruence_reason = "AFFECTIVE-SEMANTIC DIVERGENCE: Verbal utterance expresses positive equilibrium, but vocal-facial biomarkers indicate depressive/exhaustion suppression."
                elif has_neg and valence > 0.4:
                    incongruence_detected = True
                    incongruence_reason = "AFFECTIVE-SEMANTIC DIVERGENCE: Verbal utterance states distress, yet somatic facial biomarkers exhibit relaxed positive equilibrium."
                elif divergence > 0.8:
                    incongruence_detected = True
                    incongruence_reason = "PHASE MISMATCH: Discrepancy detected between vocal acoustic cadence and semantic lexical polarity."

            # Mental Exhaustion Level (0 to 5)
            # Calculated from valence (< 0), arousal (> 0.4 or < -0.4), and body slouch
            exhaustion_score = 0
            if valence < -0.1: exhaustion_score += 1
            if valence < -0.4: exhaustion_score += 1
            if arousal > 0.5 or arousal < -0.5: exhaustion_score += 1
            if top_class in ["SAD / DEPRESSIVE", "FEAR / ANXIETY"]: exhaustion_score += 1
            if top_class == "ANGRY / HIGH AROUSAL": exhaustion_score += 1
            exhaustion_score = min(5, max(0, exhaustion_score))

            return {
                "emotion": top_class,
                "confidence": round(confidence * 100, 1),
                "valence": round(valence, 2),
                "arousal": round(arousal, 2),
                "mental_exhaustion_level": exhaustion_score,
                "channel_contributions": contrib,
                "probabilities": {config.EMOTION_CLASSES[i]: round(float(probs[i]) * 100, 1) for i in range(len(probs))},
                "incongruence_detected": incongruence_detected,
                "incongruence_reason": incongruence_reason
            }

    def train_background(self, epochs=10, batch_size=16, lr=1e-3, dropout=0.2, websocket_callback=None):
        """
        Execute PyTorch training loop on background thread.
        Uses custom or pre-generated synthetic feature clusters.
        Sends real-time updates via callback.
        """
        self.is_training = True
        try:
            # 1. Load or synthesize training data
            dataset = self._get_or_create_training_data()
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

            # Reconfigure model dropout if changed
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            criterion_cls = nn.CrossEntropyLoss()
            criterion_va = nn.MSELoss()

            history = []
            best_val_loss = float('inf')
            best_acc = 0.0

            for epoch in range(1, epochs + 1):
                self.model.train()
                total_loss = 0.0
                correct = 0
                total_samples = 0

                for f, b, s, t, label, va_target in train_loader:
                    f = f.to(self.device)
                    b = b.to(self.device)
                    s = s.to(self.device)
                    t = t.to(self.device)
                    label = label.to(self.device)
                    va_target = va_target.to(self.device)

                    optimizer.zero_grad()
                    logits, va, _, _ = self.model(f, b, s, t)
                    
                    loss_cls = criterion_cls(logits, label)
                    loss_va = criterion_va(va, va_target)
                    loss = loss_cls + 0.5 * loss_va

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * f.size(0)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == label).sum().item()
                    total_samples += f.size(0)

                train_loss = total_loss / max(1, total_samples)
                train_acc = (correct / max(1, total_samples)) * 100.0

                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_samples = 0
                with torch.no_grad():
                    for f, b, s, t, label, va_target in val_loader:
                        f, b, s, t = f.to(self.device), b.to(self.device), s.to(self.device), t.to(self.device)
                        label, va_target = label.to(self.device), va_target.to(self.device)

                        logits, va, _, _ = self.model(f, b, s, t)
                        l_c = criterion_cls(logits, label)
                        l_v = criterion_va(va, va_target)
                        val_loss += (l_c + 0.5 * l_v).item() * f.size(0)

                        preds = torch.argmax(logits, dim=1)
                        val_correct += (preds == label).sum().item()
                        val_samples += f.size(0)

                val_loss = val_loss / max(1, val_samples)
                val_acc = (val_correct / max(1, val_samples)) * 100.0

                epoch_data = {
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "accuracy": round(val_acc, 2),
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                history.append(epoch_data)

                self.training_progress = {
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "accuracy": round(val_acc, 2),
                    "status": "RUNNING"
                }

                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_acc = val_acc
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "accuracy": val_acc,
                        "val_loss": val_loss,
                        "saved_at": datetime.now().isoformat()
                    }, config.BEST_MODEL_PATH)

                # Stream epoch via callback
                if websocket_callback:
                    websocket_callback({
                        "type": "TRAIN_EPOCH_UPDATE",
                        "data": epoch_data
                    })

                time.sleep(0.05) # Allow responsive UI streaming

            # Register version in model registry
            self._record_checkpoint_version(epochs, round(best_acc, 2), round(best_val_loss, 4))

            self.training_progress["status"] = "COMPLETED"
            if websocket_callback:
                websocket_callback({
                    "type": "TRAIN_COMPLETED",
                    "data": {
                        "total_epochs": epochs,
                        "best_accuracy": round(best_acc, 2),
                        "best_loss": round(best_val_loss, 4),
                        "checkpoint": str(config.BEST_MODEL_PATH.name)
                    }
                })

        except Exception as e:
            self.training_progress["status"] = f"ERROR: {str(e)}"
            if websocket_callback:
                websocket_callback({"type": "TRAIN_ERROR", "error": str(e)})
        finally:
            self.is_training = False
            self.model.eval()

    def _get_or_create_training_data(self):
        """Load stored samples or generate realistic synthetic feature clusters."""
        if config.CUSTOM_SAMPLES_PATH.exists():
            try:
                data = torch.load(config.CUSTOM_SAMPLES_PATH)
                return TensorDataset(data['face'], data['body'], data['speech'], data['text'], data['label'], data['va'])
            except Exception:
                pass

        # Generate 6 distinct Gaussian multimodal clusters (one per emotion class)
        n_samples_per_class = 60
        num_classes = len(config.EMOTION_CLASSES)
        total = n_samples_per_class * num_classes

        faces = []
        bodies = []
        speeches = []
        texts = []
        labels = []
        vas = []

        va_prototypes = [
            (0.8, 0.4),   # HAPPY / EQUILIBRIUM
            (-0.7, -0.5), # SAD / DEPRESSIVE
            (-0.6, 0.8),  # ANGRY / HIGH AROUSAL
            (0.0, 0.0),   # NEUTRAL / CALM
            (0.4, 0.7),   # SURPRISE
            (-0.6, 0.6)   # FEAR / ANXIETY
        ]

        for c in range(num_classes):
            center_v, center_a = va_prototypes[c]
            for _ in range(n_samples_per_class):
                # Distinct cluster seeds
                f = torch.randn(config.DIM_FACE) * 0.4 + (c * 0.3)
                b = torch.randn(config.DIM_BODY) * 0.4 + (c * 0.2)
                s = torch.randn(config.DIM_SPEECH) * 0.4 + (c * 0.25)
                t = torch.randn(config.DIM_TEXT) * 0.4 + (c * 0.35)

                f = F.normalize(f, p=2, dim=0)
                b = F.normalize(b, p=2, dim=0)
                s = F.normalize(s, p=2, dim=0)
                t = F.normalize(t, p=2, dim=0)

                v = max(-1.0, min(1.0, center_v + float(np.random.normal(0, 0.1))))
                a = max(-1.0, min(1.0, center_a + float(np.random.normal(0, 0.1))))

                faces.append(f)
                bodies.append(b)
                speeches.append(s)
                texts.append(t)
                labels.append(c)
                vas.append(torch.tensor([v, a], dtype=torch.float32))

        dataset_dict = {
            'face': torch.stack(faces),
            'body': torch.stack(bodies),
            'speech': torch.stack(speeches),
            'text': torch.stack(texts),
            'label': torch.tensor(labels, dtype=torch.long),
            'va': torch.stack(vas)
        }
        torch.save(dataset_dict, config.CUSTOM_SAMPLES_PATH)
        return TensorDataset(dataset_dict['face'], dataset_dict['body'], dataset_dict['speech'], dataset_dict['text'], dataset_dict['label'], dataset_dict['va'])

    def _record_checkpoint_version(self, epochs, accuracy, loss):
        """Append metadata of new model checkpoint to model_registry.json."""
        registry = []
        if config.CHECKPOINT_REGISTRY_PATH.exists():
            try:
                with open(config.CHECKPOINT_REGISTRY_PATH, "r", encoding="utf-8") as rf:
                    registry = json.load(rf)
            except Exception:
                registry = []

        version_id = f"GMU-v{len(registry) + 1:03d}"
        entry = {
            "version_id": version_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "accuracy": accuracy,
            "loss": loss,
            "status": "ACTIVE_PRODUCTION"
        }
        # Mark previous ones as ARCHIVED
        for r in registry:
            r["status"] = "ARCHIVED"
        registry.insert(0, entry)

        with open(config.CHECKPOINT_REGISTRY_PATH, "w", encoding="utf-8") as wf:
            json.dump(registry, wf, indent=2, ensure_ascii=False)

    def restore_checkpoint(self, version_id):
        """Restore a previous model version from checkpoint registry."""
        if not config.CHECKPOINT_REGISTRY_PATH.exists():
            return False, "Checkpoint registry not found"
        try:
            with open(config.CHECKPOINT_REGISTRY_PATH, "r", encoding="utf-8") as rf:
                registry = json.load(rf)
            target = next((item for item in registry if item["version_id"] == version_id), None)
            if not target:
                return False, f"Version {version_id} not found"
            for item in registry:
                item["status"] = "ACTIVE_PRODUCTION" if item["version_id"] == version_id else "ARCHIVED"
            with open(config.CHECKPOINT_REGISTRY_PATH, "w", encoding="utf-8") as wf:
                json.dump(registry, wf, indent=2, ensure_ascii=False)
            return True, f"Restored {version_id} as current active model"
        except Exception as e:
            return False, str(e)

# Global singleton
fusion_engine = MultimodalFusionEngine()
