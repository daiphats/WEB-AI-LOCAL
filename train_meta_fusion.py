"""
CMP169 - Phase 5.6: Train Meta-Learner Multimodal Fusion (Stacking Ensemble)
================================================================================
Trains a Scikit-Learn Meta-Learner (LogisticRegression / RandomForest) on top of
7-class Audio (CNN-LSTM SER) and 7-class Face (DeepFace FER with 5-frame Temporal
Aggregation) probability distributions (14-dimensional feature vector).

Saves the trained meta-model to 'checkpoints/meta_fusion_model.pkl'.

Author: Expert AI/ML Engineer
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Set UTF-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import evaluation and preprocessing utilities from Phase 5.3 & 5.5
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import evaluate_fusion_blind_test as ef
from dataset_pipeline_v2 import EMOTION_CLASSES


def extract_video_features(
    sample: Dict,
    ser_model: torch.nn.Module,
    device: torch.device
) -> Optional[Tuple[np.ndarray, np.ndarray, str, str]]:
    """
    Extracts 7-class audio_probs and 7-class face_probs for a single video.
    Returns: (audio_probs, face_probs, ground_truth, filename)
    """
    vpath = sample["file_path"]
    gt_label = sample["ground_truth"]

    try:
        # 1. Voice Modality: 16kHz mono, Mel-128, CMVN -> SER CNN-LSTM
        audio_sig = ef.extract_audio_from_video(vpath, target_sr=16000)
        audio_feat = ef.preprocess_speech_signal(audio_sig, sr=16000)
        feat_tensor = torch.from_numpy(audio_feat).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            audio_probs = ser_model(feat_tensor, return_probs=True).cpu().numpy()[0]

        # 2. Face Modality: Phase 5.5 5-frame temporal sampling -> DeepFace FER
        frames = ef.extract_temporal_frames(vpath, num_frames=5)
        face_probs = ef.compute_face_emotion_probs(frames)

        return audio_probs, face_probs, gt_label, sample["filename"]
    except Exception as e:
        print(f"[WARN] Failed extracting {sample['filename']}: {e}")
        return None


def collect_or_load_dataset(
    data_dir: Path,
    ckpt_path: Path,
    cache_path: Path,
    max_samples: Optional[int] = None,
    num_workers: int = 8,
    recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Collects 14-dim multimodal feature vectors (X) and ground truth labels (y).
    Uses caching for fast iteration.
    """
    if cache_path.exists() and not recompute:
        print(f"[CACHE] Loading cached features from '{cache_path.resolve()}'...")
        data = np.load(str(cache_path), allow_pickle=True)
        X = data["X"]
        y = data["y"]
        filenames = list(data["filenames"])
        print(f"[CACHE] Loaded {len(X)} feature vectors of dimension {X.shape[1]}.")
        return X, y, filenames

    print(f"\n[DATA DISCOVERY] Scanning videos in '{data_dir.resolve()}'...")
    all_samples = ef.discover_ravdess_video_samples(data_dir)
    if len(all_samples) == 0:
        raise ValueError(f"No valid video samples found in {data_dir}")

    # Subsample if max_samples is set (stratified across emotions)
    if max_samples and max_samples < len(all_samples):
        print(f"[SAMPLING] Selecting {max_samples} balanced samples across classes...")
        by_class: Dict[str, List[Dict]] = {c: [] for c in EMOTION_CLASSES}
        for s in all_samples:
            by_class[s["ground_truth"]].append(s)
        samples = []
        per_class = max(1, max_samples // len(EMOTION_CLASSES))
        for c in EMOTION_CLASSES:
            samples.extend(by_class[c][:per_class])
        # Add remaining to reach max_samples if needed
        remaining = [s for s in all_samples if s not in samples]
        samples.extend(remaining[:max_samples - len(samples)])
    else:
        samples = all_samples

    print(f"[DATASET] Extracting features for {len(samples)} videos across {num_workers} parallel workers...")

    # Load SER Audio Model
    device = torch.device("cpu")
    ser_model = ef.CNNLSTM_SER(num_classes=7)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        ser_model.load_state_dict(ckpt["model_state_dict"])
    else:
        ser_model.load_state_dict(ckpt)
    ser_model.eval()

    X_list = []
    y_list = []
    filenames_list = []

    start_time = time.time()

    def worker(s):
        return extract_video_features(s, ser_model, device)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(worker, samples))

    for res in results:
        if res is not None:
            a_probs, f_probs, gt, fname = res
            # Concatenate [face_probs, audio_probs] into 14-dimensional feature vector
            feature_14 = np.concatenate([f_probs, a_probs]).astype(np.float32)
            X_list.append(feature_14)
            y_list.append(gt)
            filenames_list.append(fname)

    duration = time.time() - start_time
    print(f"[EXTRACTION] Extracted {len(X_list)} feature vectors in {duration:.2f}s ({duration/len(X_list):.2f}s/video).")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), X=X, y=y, filenames=np.array(filenames_list))
    print(f"[CACHE] Saved feature cache to '{cache_path.resolve()}'.")

    return X, y, filenames_list


def main():
    parser = argparse.ArgumentParser(description="Phase 5.6: Train Meta-Learner Multimodal Fusion (Stacking Ensemble)")
    parser.add_argument("--data_dir", type=str, default="data/RAVDESS_Video_Test", help="Path to video dataset directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_ser_cnn_lstm_v2.pt", help="Path to SER CNN-LSTM checkpoint")
    parser.add_argument("--cache_file", type=str, default="checkpoints/meta_features_cache.npz", help="Path to cache extracted 14D features")
    parser.add_argument("--output_model", type=str, default="checkpoints/meta_fusion_model.pkl", help="Path to save trained meta-model")
    parser.add_argument("--model_type", type=str, default="logistic", choices=["logistic", "rf", "best"], help="Classifier type to train")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to extract (None for all)")
    parser.add_argument("--num_workers", type=int, default=8, help="Worker threads for parallel extraction")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of features")
    args = parser.parse_args()

    base_dir = _SCRIPT_DIR
    data_dir = (base_dir / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    ckpt_path = (base_dir / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    cache_path = (base_dir / args.cache_file).resolve() if not Path(args.cache_file).is_absolute() else Path(args.cache_file)
    output_model = (base_dir / args.output_model).resolve() if not Path(args.output_model).is_absolute() else Path(args.output_model)

    print("\n" + "=" * 80)
    print("   CMP169: PHASE 5.6 - TRAIN META-LEARNER MULTIMODAL FUSION ENGINE")
    print("           STACKING ENSEMBLE: 14D (FACE 7D + AUDIO 7D) -> 7 CLASSES")
    print("=" * 80)

    # 1. Collect or Load Dataset
    X, y, filenames = collect_or_load_dataset(
        data_dir=data_dir,
        ckpt_path=ckpt_path,
        cache_path=cache_path,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        recompute=args.recompute
    )

    print(f"\n[DATA SUMMARY] Total Samples: {len(X)} | Feature Dimension: {X.shape[1]}")
    for c in EMOTION_CLASSES:
        cnt = np.sum(y == c)
        print(f"   - {c.upper():<10}: {cnt} samples ({cnt/len(y)*100:.1f}%)")

    # 2. Train / Validation Split (80% Train, 20% Val, Stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[SPLIT] Training Set: {len(X_train)} samples | Validation Set: {len(X_val)} samples")

    # 3. Baseline 80/20 Decision-Level Late Fusion Performance on Validation Set
    val_face_probs = X_val[:, :7]
    val_audio_probs = X_val[:, 7:]
    baseline_fused_probs = (0.8 * val_face_probs) + (0.2 * val_audio_probs)
    baseline_preds = [EMOTION_CLASSES[i] for i in np.argmax(baseline_fused_probs, axis=1)]
    baseline_acc = accuracy_score(y_val, baseline_preds) * 100.0

    face_only_preds = [EMOTION_CLASSES[i] for i in np.argmax(val_face_probs, axis=1)]
    face_acc = accuracy_score(y_val, face_only_preds) * 100.0

    audio_only_preds = [EMOTION_CLASSES[i] for i in np.argmax(val_audio_probs, axis=1)]
    audio_acc = accuracy_score(y_val, audio_only_preds) * 100.0

    print(f"\n[BASELINE COMPARISON ON VAL SET ({len(y_val)} samples)]")
    print(f"   * Voice-Only (CNN-LSTM SER) Accuracy  : {audio_acc:.2f}%")
    print(f"   * Face-Only (DeepFace 5-Frame) Accuracy: {face_acc:.2f}%")
    print(f"   * Baseline 80/20 Late Fusion Accuracy : {baseline_acc:.2f}%")

    # 4. Train Candidates
    # Logistic Regression Meta-Learner (Convex, well-calibrated Softmax probabilities)
    lr_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_clf.fit(X_train, y_train)
    lr_train_acc = accuracy_score(y_train, lr_clf.predict(X_train)) * 100.0
    lr_val_acc = accuracy_score(y_val, lr_clf.predict(X_val)) * 100.0

    # Random Forest Meta-Learner (Non-linear modality interactions)
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_train_acc = accuracy_score(y_train, rf_clf.predict(X_train)) * 100.0
    rf_val_acc = accuracy_score(y_val, rf_clf.predict(X_val)) * 100.0

    print(f"\n[META-LEARNER TRAINING RESULTS]")
    print(f"   * Logistic Regression : Train Acc = {lr_train_acc:.2f}% | Val Acc = {lr_val_acc:.2f}%")
    print(f"   * Random Forest       : Train Acc = {rf_train_acc:.2f}% | Val Acc = {rf_val_acc:.2f}%")

    # Select Best Model
    if args.model_type == "rf":
        chosen_clf = rf_clf
        chosen_name = "RandomForestClassifier"
        chosen_val_acc = rf_val_acc
    elif args.model_type == "best":
        if rf_val_acc > lr_val_acc:
            chosen_clf = rf_clf
            chosen_name = "RandomForestClassifier"
            chosen_val_acc = rf_val_acc
        else:
            chosen_clf = lr_clf
            chosen_name = "LogisticRegression"
            chosen_val_acc = lr_val_acc
    else:
        chosen_clf = lr_clf
        chosen_name = "LogisticRegression"
        chosen_val_acc = lr_val_acc

    # Retrain on full dataset (X, y) to maximize generalization capacity
    print(f"\n[FINAL TRAINING] Fitting {chosen_name} on entire dataset ({len(X)} samples)...")
    chosen_clf.fit(X, y)
    full_preds = chosen_clf.predict(X)
    full_acc = accuracy_score(y, full_preds) * 100.0
    print(f"[FINAL TRAINING] Full Dataset Accuracy: {full_acc:.2f}%")

    # 5. Save Model Checkpoint
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(chosen_clf, str(output_model))
    print(f"\n[MODEL SAVED] Meta-Learner model successfully saved to '{output_model.resolve()}'.")

    # Verification: Verify predict_proba output shape and sum
    sample_x = X[0:1]
    sample_probs = chosen_clf.predict_proba(sample_x)[0]
    print(f"[VERIFICATION] Sample 14D prediction: shape={sample_probs.shape}, sum={sample_probs.sum():.6f}")
    assert sample_probs.shape == (7,), f"Expected shape (7,), got {sample_probs.shape}"
    assert abs(float(sample_probs.sum()) - 1.0) < 1e-4, f"Probabilities must sum to 1.0"

    print("\n" + "=" * 80)
    print("   [PHASE 5.6] META-LEARNER TRAINING COMPLETED SUCCESSFULLY!")
    print(f"   Model Checkpoint : {output_model.name}")
    print(f"   Architecture     : {chosen_name} Stacking Meta-Learner")
    print(f"   Input Dimension  : 14D (Face 7D + Audio 7D)")
    print(f"   Output Classes   : 7 Classes ({list(chosen_clf.classes_)})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

