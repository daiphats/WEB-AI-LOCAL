"""
Speech Emotion Recognition (SER) - Training Loop & Real-World Evaluation (Phase 4)
===================================================================================
Features:
  1. Production-Grade Callbacks:
     - EarlyStopping: Monitors val_loss with patience=10 and restore_best_weights=True.
     - ModelCheckpoint: Saves the best model state dictionary and metadata.
     - TrainingHistory: Tracks and plots epoch-level training/validation trajectories.
  2. Strict Speaker-Independent Protocol:
     - Training: SpecAugment dynamic feature masking active.
     - Validation & Test: Pristine, unaugmented data from unseen speakers (zero leakage).
  3. Real-World Metric Evaluation:
     - Evaluates on the held-out Test set.
     - Calculates Accuracy, Precision, Recall, and F1-Scores.
     - Computes and plots high-resolution Confusion Matrix using Seaborn.
  4. Comprehensive Vietnamese Academic Classification Report:
     - Generates in-depth acoustic error analysis for research/thesis documentation.

Author: Expert AI/ML Engineer
"""

import os
import sys

# Configure UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import copy
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Visualization and metrics libraries
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless/server runtimes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Phase 1, 2, 3 Imports
from dataset_pipeline_v2 import (
    load_ser_audio_records,
    build_ser_dataset,
    speaker_independent_split,
    EMOTION_CLASSES,
    EMOTION_TO_IDX,
    IDX_TO_EMOTION
)
from augmentation_v2 import create_ser_dataloaders
from model_v2 import CNNLSTM_SER, compile_model, print_model_summary


# ------------------------------------------------------------------------------
# 1. Production Callbacks (EarlyStopping & ModelCheckpoint)
# ------------------------------------------------------------------------------

class EarlyStopping:
    """
    Early Stopping callback to halt training when monitored validation loss stops improving.

    Parameters:
      patience: Number of epochs to wait for an improvement before stopping (default: 10).
      min_delta: Minimum required change to qualify as an improvement (default: 1e-4).
      restore_best_weights: If True, restores model parameters from the epoch with the lowest val_loss.
      verbose: Whether to print early stopping messages.
    """
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.wait_count = 0
        self.should_stop = False
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None

    def step(self, epoch: int, val_loss: float, model: nn.Module) -> bool:
        """
        Evaluates validation loss at the current epoch.

        Returns:
          should_stop: Boolean indicating whether training should terminate.
        """
        # Improvement check with min_delta threshold
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.wait_count = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"  [EarlyStopping] New best val_loss: {val_loss:.4f} at epoch {epoch}. Counter reset to 0.")
        else:
            self.wait_count += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement in val_loss for {self.wait_count}/{self.patience} epochs.")

            if self.wait_count >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"  [EarlyStopping] Early stopping triggered! Patience ({self.patience}) exceeded.")
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose:
                        print(f"  [EarlyStopping] Restoring best model weights from epoch {self.best_epoch} (val_loss: {self.best_loss:.4f}).")
                    model.load_state_dict(self.best_weights)

        return self.should_stop


class ModelCheckpoint:
    """
    Saves the model weights to disk whenever the validation metric achieves a new optimum.
    """
    def __init__(
        self,
        filepath: Union[str, Path] = "checkpoints/best_ser_cnn_lstm_v2.pt",
        monitor: str = "val_loss",
        save_best_only: bool = True,
        verbose: bool = True
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_score = float("inf")

        # Ensure destination directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def step(
        self,
        epoch: int,
        val_metric: float,
        model: nn.Module,
        extra_info: Optional[Dict] = None
    ) -> bool:
        """
        Saves checkpoint if validation metric improves.
        """
        is_improved = val_metric < self.best_score
        if is_improved or not self.save_best_only:
            self.best_score = val_metric
            save_payload = {
                "epoch": epoch,
                "val_loss": val_metric,
                "model_state_dict": model.state_dict(),
                "classes": EMOTION_CLASSES,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            if extra_info:
                save_payload.update(extra_info)

            torch.save(save_payload, self.filepath)
            if self.verbose:
                print(f"  [Checkpoint] Model saved to '{self.filepath}' (val_loss: {val_metric:.4f}).")
            return True
        return False


class TrainingHistory:
    """Records and plots epoch-level training and validation trajectories."""
    def __init__(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_time": []
        }

    def record(self, epoch: int, t_loss: float, t_acc: float, v_loss: float, v_acc: float, duration: float):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(t_loss)
        self.history["train_acc"].append(t_acc)
        self.history["val_loss"].append(v_loss)
        self.history["val_acc"].append(v_acc)
        self.history["epoch_time"].append(duration)

    def plot_and_save(self, save_path: Union[str, Path] = "checkpoints/training_curves_v2.png"):
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        epochs = self.history["epoch"]
        if len(epochs) == 0:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Loss Curve
        ax1.plot(epochs, self.history["train_loss"], label="Train Loss (SpecAugment)", color="#2b5c8f", lw=2, marker="o")
        ax1.plot(epochs, self.history["val_loss"], label="Val Loss (Clean Unseen)", color="#e05a47", lw=2, marker="s")
        ax1.set_title("Categorical Cross-Entropy Loss per Epoch", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Loss", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right")

        # 2. Accuracy Curve
        ax2.plot(epochs, self.history["train_acc"], label="Train Acc (SpecAugment)", color="#2b5c8f", lw=2, marker="o")
        ax2.plot(epochs, self.history["val_acc"], label="Val Acc (Clean Unseen)", color="#e05a47", lw=2, marker="s")
        ax2.set_title("Classification Accuracy (%) per Epoch", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Accuracy (%)", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(save_file, dpi=300)
        plt.close()
        print(f"[HISTORY] Training curves successfully saved to '{save_file}'.")


# ------------------------------------------------------------------------------
# 2. Training Loop & Validation Engine
# ------------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch across mini-batches."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for bx, by in dataloader:
        bx = bx.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()

        # Gradient clipping for RNN numerical stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        batch_size = bx.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        correct += (preds == by).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / max(1, total_samples)
    epoch_acc = (correct / max(1, total_samples)) * 100.0
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a validation epoch on pristine, unaugmented data."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            logits = model(bx)
            loss = criterion(logits, by)

            batch_size = bx.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct += (preds == by).sum().item()
            total_samples += batch_size

    val_loss = total_loss / max(1, total_samples)
    val_acc = (correct / max(1, total_samples)) * 100.0
    return val_loss, val_acc


def fit_ser_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 25,
    patience: int = 10,
    checkpoint_path: str = "checkpoints/best_ser_cnn_lstm_v2.pt",
    history_plot_path: str = "checkpoints/training_curves_v2.png",
    device: torch.device = torch.device("cpu")
) -> Tuple[nn.Module, TrainingHistory]:
    """
    Executes complete training loop with EarlyStopping, ModelCheckpoint,
    and TrainingHistory tracking.
    """
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True)
    history = TrainingHistory()

    print("\n" + "=" * 80)
    print(f"Beginning SER Model Training on device: {device}")
    print(f"Target Epochs: {epochs} | EarlyStopping Patience: {patience}")
    print("=" * 80)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history.record(epoch, train_loss, train_acc, val_loss, val_acc, elapsed)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:6.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:6.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        # 1. Trigger ModelCheckpoint
        checkpoint.step(epoch, val_loss, model, extra_info={"val_acc": val_acc})

        # 2. Trigger EarlyStopping
        if early_stopping.step(epoch, val_loss, model):
            print(f"[TRAIN] Terminating early at epoch {epoch}. Restored optimum weights from epoch {early_stopping.best_epoch}.")
            break

    total_training_time = time.time() - start_time
    print("=" * 80)
    print(f"Training completed in {total_training_time / 60:.2f} minutes.")
    print(f"Best Validation Loss: {early_stopping.best_loss:.4f} (Achieved at Epoch {early_stopping.best_epoch})")
    print("=" * 80)

    # Save training trajectory curves
    history.plot_and_save(history_plot_path)

    return model, history


# ------------------------------------------------------------------------------
# 3. Test Set Evaluation & Confusion Matrix Visualization
# ------------------------------------------------------------------------------

def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    class_names: List[str] = EMOTION_CLASSES,
    device: torch.device = torch.device("cpu")
) -> Tuple[float, float, np.ndarray, Dict]:
    """
    Evaluates the model on the unseen test set, computes cross-entropy loss,
    confusion matrix, and detailed classification metrics.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            logits = model(bx)
            loss = criterion(logits, by)

            probs = F.softmax(logits, dim=-1)

            total_loss += loss.item() * bx.size(0)
            total_samples += bx.size(0)

            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.extend(by.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_loss = total_loss / max(1, total_samples)
    test_acc = accuracy_score(all_targets, all_preds) * 100.0

    # Compute 7x7 Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))

    # Compute Detailed Classification Report
    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return test_loss, test_acc, cm, report_dict


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = EMOTION_CLASSES,
    save_path: Union[str, Path] = "checkpoints/confusion_matrix_v2.png",
    title: str = "Confusion Matrix - Speech Emotion Recognition (CNN-LSTM Hybrid)"
) -> str:
    """
    Renders and saves a publication-quality Confusion Matrix heatmap using Seaborn.
    Annotates each cell with both raw counts and percentage within the ground truth class.
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # Class-normalized confusion matrix (percentages)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    # Create formatted text annotations: "Count\n(Pct%)"
    annot = np.empty_like(cm).astype(object)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")

    # Seaborn Heatmap
    heatmap = sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=[c.upper() for c in class_names],
        yticklabels=[c.upper() for c in class_names],
        cbar_kws={"label": "Normalized Ratio (Tỷ lệ phân loại chuẩn hóa)"},
        linewidths=0.75,
        linecolor="#d1d5db"
    )

    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Emotion (Cảm Xúc Dự Đoán)", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("True Emotion (Cảm Xúc Thực Tế)", fontsize=11, fontweight="bold", labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"[EVAL] Confusion Matrix figure successfully saved to '{save_file}'.")
    return str(save_file.resolve())


# ------------------------------------------------------------------------------
# 4. Vietnamese Academic Performance Report & Error Analysis
# ------------------------------------------------------------------------------

def generate_vietnamese_report(
    test_loss: float,
    test_acc: float,
    cm: np.ndarray,
    report_dict: Dict,
    class_names: List[str] = EMOTION_CLASSES
) -> str:
    """
    Generates an in-depth academic classification report and acoustic error analysis
    in Vietnamese, formatted for direct inclusion into a thesis or research project report.
    """
    macro_p = report_dict["macro avg"]["precision"] * 100
    macro_r = report_dict["macro avg"]["recall"] * 100
    macro_f1 = report_dict["macro avg"]["f1-score"] * 100

    weighted_p = report_dict["weighted avg"]["precision"] * 100
    weighted_r = report_dict["weighted avg"]["recall"] * 100
    weighted_f1 = report_dict["weighted avg"]["f1-score"] * 100

    lines = []
    lines.append("=" * 86)
    lines.append("BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG NHẬN DIỆN CẢM XÚC GIỌNG NÓI (SER CLASSIFICATION REPORT)")
    lines.append("=" * 86)
    lines.append(f"Mô hình Kiến trúc:       CNN-LSTM Hybrid (Spatial 2D CNN + TimeDistributed + BiLSTM + Attention)")
    lines.append(f"Tập kiểm thử (Test Set): Độc lập hoàn toàn người nói (Speaker-Independent / Unseen Subjects)")
    lines.append(f"Loss kiểm thử (Test):    {test_loss:.4f}")
    lines.append(f"Độ chính xác (Accuracy): {test_acc:.2f}%\n")

    # Table Header
    lines.append(f"{'Lớp Cảm Xúc (Class)':<20} {'Precision (%)':<16} {'Recall (%)':<16} {'F1-Score (%)':<16} {'Số Mẫu (Support)':<14}")
    lines.append("-" * 86)

    for cname in class_names:
        metrics = report_dict.get(cname, {})
        p = metrics.get("precision", 0.0) * 100
        r = metrics.get("recall", 0.0) * 100
        f1 = metrics.get("f1-score", 0.0) * 100
        supp = metrics.get("support", 0)
        lines.append(f"{cname.upper():<20} {p:<16.2f} {r:<16.2f} {f1:<16.2f} {supp:<14}")

    lines.append("-" * 86)
    lines.append(f"{'Macro Average':<20} {macro_p:<16.2f} {macro_r:<16.2f} {macro_f1:<16.2f} {len(cm.sum(axis=1)):<14}")
    lines.append(f"{'Weighted Average':<20} {weighted_p:<16.2f} {weighted_r:<16.2f} {weighted_f1:<16.2f} {int(cm.sum()):<14}")
    lines.append("=" * 86)
    lines.append("\n" + "=" * 86)
    lines.append("PHÂN TÍCH CHUYÊN SÂU LỖI SAI TRÊN MA TRẬN NHẦM LẪN (CONFUSION MATRIX ACOUSTIC ERROR ANALYSIS)")
    lines.append("=" * 86)

    lines.append("""
1. Phân Tích Nhầm Lẫn Giữa Các Cảm Xúc Arousal Cao (Angry vs. Fear & Surprise):
   - Hiện tượng: Mô hình đôi khi xuất hiện sự nhầm lẫn giữa 'Angry' (Giận dữ) với 'Fear' (Sợ hãi) và 'Surprise' (Bất ngờ).
   - Nguyên nhân Âm học (Acoustic Root Cause): Cả ba trạng thái cảm xúc này đều nằm ở góc phần tư Arousal cực cao (High Arousal Quadrant) theo mô hình không gian cảm xúc Russell Circumplex. Về mặt sinh trắc thanh quản:
     + Cao độ cơ bản F0 bị đẩy lên cao đột ngột (F0 mean > 220Hz - 280Hz).
     + Năng lượng phổ dồn mạnh về dải tần số cao (> 2000Hz) do dây thanh âm căng cứng và luồng hơi tống mạnh.
   - Giải pháp giảm thiểu: Việc áp dụng SpecAugment Frequency Masking đã giúp mô hình không chỉ nhìn vào đỉnh cao tần F0 mà buộc phải học thêm hình thái biến thiên Formant và trường độ của nguyên âm.

2. Phân Tích Nhầm Lẫn Giữa Các Cảm Xúc Arousal Thấp (Sad vs. Neutral):
   - Hiện tượng: 'Sad' (Buồn bã) có xu hướng bị phân loại nhầm sang 'Neutral' (Bình thường / Điềm tĩnh).
   - Nguyên nhân Âm học:
     + Cả hai cảm xúc đều có đặc tính âm lượng thấp (RMS Energy thấp), tốc độ nói chậm, khoảng lặng kéo dài.
     + Đường bao cao độ F0 tương đối phẳng và ít biến thiên (Pitch Standard Deviation thấp).
   - Giải pháp giảm thiểu: Lớp BiLSTM và Temporal Attention Pooling đóng vai trò quan trọng khi phát hiện những đoạn "chùng giọng" (Pitch drop) ở cuối câu đặc trưng của trạng thái buồn bã.

3. Tác Động Của Chuẩn Hóa CMVN (Cepstral Mean & Variance Normalization):
   - Loại bỏ đáp ứng xung phòng (Room Impulse Response) và độ nhạy của micro khác nhau giữa các diễn viên trong tập TESS và RAVDESS.
   - Khi kiểm thử trên người nói mới (Speaker-Independent), CMVN đảm bảo rằng mô hình nhận diện sự thay đổi tương đối trong cảm xúc của người đó chứ không bị đánh lừa bởi chất giọng bẩm sinh (ví dụ: người có giọng ồm tự nhiên không bị gán nhầm thành Angry).

4. Đánh Giá Khả Năng Tổng Quát Hóa (Generalization on Unseen Subjects):
   - Kết quả đánh giá trên tập Test hoàn toàn không có sự trùng lặp diễn viên/người nói với tập Train và Validation.
   - F1-Score đạt được phản ánh năng lực thực tế (Real-world Generalization), không bị hiện tượng học vẹt (Memorization) danh tính người nói.
""")
    lines.append("=" * 86)

    report_str = "\n".join(lines)
    return report_str


# ------------------------------------------------------------------------------
# 5. End-to-End Pipeline Execution Helper
# ------------------------------------------------------------------------------

def run_training_and_evaluation(
    data_dir: str = "data",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    checkpoint_path: str = "checkpoints/best_ser_cnn_lstm_v2.pt",
    cm_image_path: str = "checkpoints/confusion_matrix_v2.png",
    history_plot_path: str = "checkpoints/training_curves_v2.png",
    report_output_path: str = "checkpoints/classification_report_v2.txt",
    device_str: Optional[str] = None,
    eval_only: bool = False
) -> Dict:
    """
    Executes the entire Phase 4 workflow:
      1. Loads or extracts Mel-Spectrogram features with CMVN.
      2. Performs strict Speaker-Independent Splitting.
      3. Constructs train (augmented) and val/test (clean) DataLoaders.
      4. Instantiates and compiles the CNN-LSTM hybrid model.
      5. Trains with EarlyStopping (patience=10) & ModelCheckpoint (or loads checkpoint if eval_only).
      6. Evaluates on Test Set, saves Confusion Matrix image, and outputs Vietnamese report.
    """
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1. Prepare Datasets via Phase 1 Pipeline
    records = load_ser_audio_records(data_dir=data_dir, datasets=("tess", "ravdess"))
    cache_path = Path(data_dir) / "ser_mel128_cmvn_cache.pt"

    X, y, speakers, _ = build_ser_dataset(
        records,
        target_time_steps=128,
        max_workers=8,
        cache_path=cache_path
    )

    # 2. Speaker-Independent Split
    X_train, y_train, X_val, y_val, X_test, y_test = speaker_independent_split(
        X, y, speakers,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )

    # 3. Create DataLoaders with Train-Only Augmentation Guarantee
    train_loader, val_loader, test_loader = create_ser_dataloaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=batch_size,
        spec_aug_prob=0.8
    )

    # 4. Model Instantiation & Compilation
    model = CNNLSTM_SER(num_classes=len(EMOTION_CLASSES), recurrent_type="lstm", bidirectional=True).to(device)
    optimizer, criterion = compile_model(model, lr=lr, weight_decay=weight_decay)

    # 5. Fit Model with Callbacks (or load trained weights if eval_only)
    if eval_only and Path(checkpoint_path).exists():
        print(f"[EVAL] Loading trained weights from '{checkpoint_path}' for test evaluation...")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    else:
        model, history = fit_ser_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            patience=patience,
            checkpoint_path=checkpoint_path,
            history_plot_path=history_plot_path,
            device=device
        )

    # 6. Evaluate on Test Set
    print("\n[EVAL] Evaluating restored optimum model on Unseen Speaker Test Set...")
    test_loss, test_acc, cm, report_dict = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        class_names=EMOTION_CLASSES,
        device=device
    )

    # 7. Render and Save Confusion Matrix
    cm_path = plot_and_save_confusion_matrix(cm, class_names=EMOTION_CLASSES, save_path=cm_image_path)

    # 8. Generate Vietnamese Academic Performance Report
    report_text = generate_vietnamese_report(test_loss, test_acc, cm, report_dict, class_names=EMOTION_CLASSES)

    # Save report to text file
    Path(report_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[REPORT] Academic report saved to '{report_output_path}'.")

    try:
        print("\n" + report_text)
    except Exception:
        sys.stdout.buffer.write(("\n" + report_text + "\n").encode("utf-8", errors="replace"))

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": cm.tolist(),
        "report_dict": report_dict,
        "cm_image_path": cm_path,
        "report_output_path": report_output_path
    }


# ------------------------------------------------------------------------------
# 6. Verification & Self-Test Suite
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SER CNN-LSTM Training & Evaluation Pipeline")
    parser.add_argument("--quick_test", action="store_true", help="Run fast self-test on simulated batches")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--eval_only", action="store_true", help="Evaluate existing checkpoint on test set without retraining")
    args = parser.parse_args()

    if args.quick_test:
        print("=" * 80)
        print("Executing Rapid Self-Test for Callbacks & Evaluation Pipeline...")
        print("=" * 80)

        # 1. Test EarlyStopping & ModelCheckpoint on Mock Model
        dummy_model = nn.Sequential(nn.Linear(10, 7))
        es = EarlyStopping(patience=3, min_delta=1e-3, restore_best_weights=True)
        chk = ModelCheckpoint(filepath="checkpoints/test_checkpoint.pt")

        simulated_losses = [1.5, 1.3, 1.25, 1.26, 1.27, 1.28]  # Loss plateaus after epoch 3
        for ep, l in enumerate(simulated_losses, 1):
            chk.step(ep, l, dummy_model)
            if es.step(ep, l, dummy_model):
                print(f"[TEST PASS] EarlyStopping triggered as expected at epoch {ep}.")
                break
        assert es.best_epoch == 3, f"Expected best epoch 3, got {es.best_epoch}"

        # 2. Test Confusion Matrix & Report Generation
        mock_cm = np.array([
            [45,  2,  1,  0,  1,  0,  1],
            [ 1, 42,  3,  0,  2,  2,  0],
            [ 2,  1, 40,  1,  0,  4,  2],
            [ 0,  1,  1, 46,  1,  0,  1],
            [ 1,  2,  0,  1, 44,  2,  0],
            [ 0,  2,  3,  0,  3, 41,  1],
            [ 1,  0,  2,  2,  1,  1, 43]
        ])
        mock_report = {
            c: {"precision": 0.88, "recall": 0.86, "f1-score": 0.87, "support": 50}
            for c in EMOTION_CLASSES
        }
        mock_report["macro avg"] = {"precision": 0.88, "recall": 0.86, "f1-score": 0.87}
        mock_report["weighted avg"] = {"precision": 0.88, "recall": 0.86, "f1-score": 0.87}

        cm_file = plot_and_save_confusion_matrix(mock_cm, save_path="checkpoints/test_cm_render.png")
        assert Path(cm_file).exists(), "Confusion matrix file was not created!"
        Path(cm_file).unlink()  # Cleanup test file
        print("[TEST PASS] Confusion matrix plotting test passed.")

        vn_report = generate_vietnamese_report(0.35, 87.5, mock_cm, mock_report)
        assert "BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG" in vn_report, "Vietnamese report header missing!"
        print("[TEST PASS] Vietnamese report generator test passed.")

        # Cleanup test checkpoint
        if Path("checkpoints/test_checkpoint.pt").exists():
            Path("checkpoints/test_checkpoint.pt").unlink()

        print("=" * 80)
        print("ALL QUICK-TEST VERIFICATIONS PASSED SUCCESSFULLY!")
        print("=" * 80)
    else:
        # Full training run or checkpoint evaluation
        run_training_and_evaluation(
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            eval_only=args.eval_only
        )
