"""
Speech Emotion Recognition (SER) - Dynamic Data Augmentation Pipeline (Phase 2)
================================================================================
Features:
  1. Dynamic Waveform Augmentation:
     - Gaussian White Noise with precise Signal-to-Noise Ratio (SNR 10 dB - 30 dB).
     - Pitch Shifting within [-2, +2] semitones without modifying tempo.
  2. SpecAugment (Frequency & Time Masking):
     - Applied directly on 128-band Mel-Spectrogram batches or individual tensors.
     - Operates natively in PyTorch (GPU/CPU accelerated, zero external STFT overhead).
  3. Safe Dataset & DataLoader Pipeline:
     - Enforces that data augmentation is applied ONLY to the training set.
     - Validation and test splits remain completely pristine and unaugmented.

Author: Expert AI/ML Engineer
"""

import os
import sys
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import Phase 1 Preprocessing primitives if available
try:
    from dataset_pipeline_v2 import extract_mel_spectrogram, apply_cmvn, EMOTION_CLASSES
except ImportError:
    # Standalone fallback definition
    EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ------------------------------------------------------------------------------
# 1. Waveform Domain Augmentations (Audio Time-Domain)
# ------------------------------------------------------------------------------

def add_gaussian_white_noise(
    waveform: np.ndarray,
    min_snr_db: float = 10.0,
    max_snr_db: float = 30.0,
    clip: bool = True,
    eps: float = 1e-9
) -> Tuple[np.ndarray, float]:
    """
    Injects dynamic Gaussian White Noise into raw 1D audio waveform at a target SNR.

    Mathematical Formulation:
      SNR_dB = 10 * log10( P_signal / P_noise )
      => P_noise = P_signal / 10^(SNR_dB / 10)
      => sigma_noise = sqrt( P_noise )
      => noise ~ N(0, sigma_noise^2)
      => y_noisy = waveform + noise

    Parameters:
      waveform: 1D numpy array of audio samples (float32).
      min_snr_db: Lower bound for Signal-to-Noise Ratio (default: 10.0 dB).
      max_snr_db: Upper bound for Signal-to-Noise Ratio (default: 30.0 dB).
      clip: Whether to clip output into [-1.0, 1.0] audio range.
      eps: Numerical floor to prevent log(0) on silent audio.

    Returns:
      noisy_waveform: Augmented audio array with same length and shape.
      target_snr_db: Sampled SNR in decibels.
    """
    sig_power = float(np.mean(waveform ** 2))
    if sig_power < eps:
        # Near-silent signal, return original to avoid excessive noise amplification
        return waveform.copy(), max_snr_db

    target_snr_db = float(np.random.uniform(min_snr_db, max_snr_db))
    noise_power = sig_power / (10.0 ** (target_snr_db / 10.0))
    noise_sigma = np.sqrt(noise_power)

    noise = np.random.normal(0.0, noise_sigma, size=waveform.shape).astype(np.float32)
    noisy_waveform = waveform + noise

    # Clip to prevent digital clipping/distortion while preserving amplitude
    if clip:
        noisy_waveform = np.clip(noisy_waveform, -1.0, 1.0)

    return noisy_waveform.astype(np.float32), target_snr_db


def pitch_shift_audio(
    waveform: np.ndarray,
    sr: int = 16000,
    min_semitones: float = -2.0,
    max_semitones: float = 2.0
) -> Tuple[np.ndarray, float]:
    """
    Shifts the vocal pitch of the waveform by [-2.0, +2.0] semitones without modifying tempo.

    Simulates physiological vocal tract pitch variations across different speakers
    and emotional arousal states.

    Parameters:
      waveform: 1D numpy array of audio samples (float32).
      sr: Audio sampling rate (default: 16000 Hz).
      min_semitones: Minimum pitch shift in semitones (default: -2.0).
      max_semitones: Maximum pitch shift in semitones (default: +2.0).

    Returns:
      shifted_waveform: Pitch-shifted audio array.
      n_steps: Sampled semitone displacement.
    """
    n_steps = float(np.random.uniform(min_semitones, max_semitones))
    # librosa.effects.pitch_shift uses phase vocoder
    shifted = librosa.effects.pitch_shift(
        y=waveform,
        sr=sr,
        n_steps=n_steps
    )
    return shifted.astype(np.float32), n_steps


class WaveformAugmenter:
    """
    Pipeline orchestrator for dynamic time-domain waveform transformations.
    Applies Gaussian White Noise (SNR 10-30dB) and Pitch Shifting (-2 to +2 semitones).
    """
    def __init__(
        self,
        sr: int = 16000,
        noise_prob: float = 0.5,
        min_snr_db: float = 10.0,
        max_snr_db: float = 30.0,
        pitch_prob: float = 0.5,
        min_semitones: float = -2.0,
        max_semitones: float = 2.0
    ):
        self.sr = sr
        self.noise_prob = noise_prob
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.pitch_prob = pitch_prob
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        out = waveform.copy()

        # 1. Dynamic Pitch Shifting
        if random.random() < self.pitch_prob:
            out, _ = pitch_shift_audio(
                out,
                sr=self.sr,
                min_semitones=self.min_semitones,
                max_semitones=self.max_semitones
            )

        # 2. Dynamic Gaussian White Noise (SNR 10dB - 30dB)
        if random.random() < self.noise_prob:
            out, _ = add_gaussian_white_noise(
                out,
                min_snr_db=self.min_snr_db,
                max_snr_db=self.max_snr_db
            )

        return out


# ------------------------------------------------------------------------------
# 2. SpecAugment (Frequency Masking & Time Masking on Mel-Spectrogram Batches)
# ------------------------------------------------------------------------------

class SpecAugment(nn.Module):
    """
    SpecAugment (Park et al., Google Brain, 2019) applied directly to Mel-Spectrogram batches.

    Mechanisms:
      1. Frequency Masking: Che dải f tần số liên tục [f0, f0 + f) với f ~ U(0, freq_mask_param).
      2. Time Masking: Che dải t khung thời gian liên tục [t0, t0 + t) với t ~ U(0, time_mask_param).

    Tensor Shapes Supported:
      - 2D: (F, T)
      - 3D: (C, F, T) or (B, F, T)
      - 4D: (B, C, F, T) [Standard mini-batch for 2D ConvNets]

    Crucial PyTorch Integration:
      - Inherits from nn.Module: automatically participates in model.train() and model.eval().
      - In eval() mode, forward() behaves as identity (0% masking).
      - GPU-accelerated: operates directly on PyTorch Tensors.
    """
    def __init__(
        self,
        freq_mask_param: int = 16,
        time_mask_param: int = 24,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        mask_value: float = 0.0,
        p: float = 0.8
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.mask_value = mask_value
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SpecAugment masking directly on batch tensor.

        Parameters:
          x: Input Mel-Spectrogram tensor of shape (B, C, F, T) or (B, F, T) or (F, T).

        Returns:
          Masked tensor with identical shape and dtype.
        """
        # If in evaluation mode or random probability gate not met, bypass augmentation
        if not self.training or (self.p < 1.0 and random.random() > self.p):
            return x

        orig_ndim = x.ndim
        # Reshape to standard 4D format (B, C, F, T)
        if orig_ndim == 2:
            # (F, T) -> (1, 1, F, T)
            x_4d = x.unsqueeze(0).unsqueeze(0)
        elif orig_ndim == 3:
            # (B, F, T) -> (B, 1, F, T)
            x_4d = x.unsqueeze(1)
        elif orig_ndim == 4:
            x_4d = x
        else:
            raise ValueError(f"SpecAugment requires 2D, 3D, or 4D tensor. Got {x.shape}")

        B, C, F, T = x_4d.shape
        x_aug = x_4d.clone()

        # Apply Frequency and Time Masking per item in the batch
        for b in range(B):
            # 1. Frequency Masking
            for _ in range(self.n_freq_masks):
                f_width = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
                if f_width > 0 and F > f_width:
                    f_start = int(torch.randint(0, F - f_width, (1,)).item())
                    x_aug[b, :, f_start : f_start + f_width, :] = self.mask_value

            # 2. Time Masking
            for _ in range(self.n_time_masks):
                t_width = int(torch.randint(0, self.time_mask_param + 1, (1,)).item())
                if t_width > 0 and T > t_width:
                    t_start = int(torch.randint(0, T - t_width, (1,)).item())
                    x_aug[b, :, :, t_start : t_start + t_width] = self.mask_value

        # Restore original tensor dimensions
        if orig_ndim == 2:
            return x_aug.squeeze(0).squeeze(0)
        elif orig_ndim == 3:
            return x_aug.squeeze(1)
        return x_aug


# ------------------------------------------------------------------------------
# 3. Dynamic PyTorch Datasets with Train-Only Augmentation Guarantee
# ------------------------------------------------------------------------------

class AugmentedSpectrogramDataset(Dataset):
    """
    PyTorch Dataset wrapper for pre-extracted Mel-Spectrogram features (e.g. from Phase 1).

    Enforces that SpecAugment is executed DYNAMICALLY on-the-fly *only* when augment=True (Train).
    Validation and test datasets strictly set augment=False.
    """
    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        augment: bool = False,
        freq_mask_param: int = 16,
        time_mask_param: int = 24,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        spec_aug_prob: float = 0.8
    ):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()

        # Ensure channel dimension: (N, 128, T) -> (N, 1, 128, T)
        if X.ndim == 3:
            X = X.unsqueeze(1)

        self.X = X
        self.y = y
        self.augment = augment

        if self.augment:
            self.spec_augmenter = SpecAugment(
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
                n_freq_masks=n_freq_masks,
                n_time_masks=n_time_masks,
                mask_value=0.0,
                p=spec_aug_prob
            )
            self.spec_augmenter.train()
        else:
            self.spec_augmenter = None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.X[idx]  # Shape: (1, 128, T)
        label = self.y[idx]

        # Apply SpecAugment strictly in training mode
        if self.augment and self.spec_augmenter is not None:
            feat = self.spec_augmenter(feat)

        return feat, label


class DynamicAudioSERDataset(Dataset):
    """
    End-to-end PyTorch Dataset for on-the-fly raw audio transformations + Mel-Spectrogram + CMVN.

    Training split (augment=True):
      1. Loads raw audio.
      2. Dynamic Pitch Shifting (-2 to +2 semitones, p=0.5).
      3. Dynamic Gaussian White Noise (SNR 10dB - 30dB, p=0.5).
      4. 128 Mel-Spectrogram + CMVN normalization.
      5. SpecAugment (Frequency & Time Masking, p=0.8).

    Val/Test split (augment=False):
      1. Loads raw audio cleanly.
      2. 128 Mel-Spectrogram + CMVN normalization.
      3. Zero noise, zero pitch shift, zero SpecAugment (completely unaugmented).
    """
    def __init__(
        self,
        records: List[Dict[str, Union[str, int]]],
        augment: bool = False,
        sr: int = 16000,
        target_time_steps: int = 128,
        noise_prob: float = 0.5,
        pitch_prob: float = 0.5,
        spec_prob: float = 0.8
    ):
        self.records = records
        self.augment = augment
        self.sr = sr
        self.target_time_steps = target_time_steps

        if self.augment:
            self.waveform_augmenter = WaveformAugmenter(
                sr=sr,
                noise_prob=noise_prob,
                min_snr_db=10.0,
                max_snr_db=30.0,
                pitch_prob=pitch_prob,
                min_semitones=-2.0,
                max_semitones=2.0
            )
            self.spec_augmenter = SpecAugment(
                freq_mask_param=16,
                time_mask_param=24,
                n_freq_masks=2,
                n_time_masks=2,
                p=spec_prob
            )
            self.spec_augmenter.train()
        else:
            self.waveform_augmenter = None
            self.spec_augmenter = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.records[idx]
        file_path = item["file_path"]
        label = int(item["label"])

        # 1. Load raw audio waveform at 16kHz
        try:
            waveform, _ = librosa.load(file_path, sr=self.sr, mono=True)
        except Exception:
            # Fallback synthetic buffer in case of rare OS I/O error
            waveform = np.zeros(int(self.sr * 2.0), dtype=np.float32)

        # 2. Dynamic waveform augmentations (Training only)
        if self.augment and self.waveform_augmenter is not None:
            waveform = self.waveform_augmenter(waveform)

        # 3. Extract 128-band Mel-Spectrogram with CMVN
        mel = extract_mel_spectrogram(
            waveform,
            sr=self.sr,
            n_mels=128,
            n_fft=1024,
            hop_length=512,
            target_time_steps=self.target_time_steps,
            apply_cmvn_flag=True
        )

        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0)  # Shape: (1, 128, target_time_steps)

        # 4. SpecAugment (Training only)
        if self.augment and self.spec_augmenter is not None:
            mel_tensor = self.spec_augmenter(mel_tensor)

        return mel_tensor, torch.tensor(label, dtype=torch.long)


# ------------------------------------------------------------------------------
# 4. DataLoader Factory Function
# ------------------------------------------------------------------------------

def create_ser_dataloaders(
    X_train: Union[np.ndarray, torch.Tensor],
    y_train: Union[np.ndarray, torch.Tensor],
    X_val: Union[np.ndarray, torch.Tensor],
    y_val: Union[np.ndarray, torch.Tensor],
    X_test: Union[np.ndarray, torch.Tensor],
    y_test: Union[np.ndarray, torch.Tensor],
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    spec_aug_prob: float = 0.8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds standard PyTorch DataLoaders for Training, Validation, and Test sets.

    Crucial Invariant Guaranteed:
      - Train Loader: augment=True, shuffle=True (SpecAugment active).
      - Val Loader:   augment=False, shuffle=False (Strictly clean).
      - Test Loader:  augment=False, shuffle=False (Strictly clean).

    Returns:
      train_loader, val_loader, test_loader
    """
    train_ds = AugmentedSpectrogramDataset(X_train, y_train, augment=True, spec_aug_prob=spec_aug_prob)
    val_ds = AugmentedSpectrogramDataset(X_val, y_val, augment=False)
    test_ds = AugmentedSpectrogramDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print("[AUGMENT] DataLoaders successfully created:")
    print(f"  Train: {len(train_ds)} samples | Batches: {len(train_loader)} | Augmentation: ACTIVE")
    print(f"  Val:   {len(val_ds)} samples | Batches: {len(val_loader)} | Augmentation: DISABLED (Clean)")
    print(f"  Test:  {len(test_ds)} samples | Batches: {len(test_loader)} | Augmentation: DISABLED (Clean)")

    return train_loader, val_loader, test_loader


# ------------------------------------------------------------------------------
# 5. Verification & Self-Test Suite
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SER Dynamic Data Augmentation Pipeline Self-Test (Phase 2)")
    print("=" * 70)

    # 1. Test Gaussian White Noise SNR Invariant
    test_sig = (np.sin(np.linspace(0, 50, 16000)) * 0.6).astype(np.float32)
    noisy_sig, target_snr = add_gaussian_white_noise(test_sig, min_snr_db=15.0, max_snr_db=25.0, clip=False)
    noise_component = noisy_sig - test_sig
    measured_snr = 10.0 * np.log10(np.mean(test_sig ** 2) / np.mean(noise_component ** 2))
    print(f"[TEST 1] Gaussian White Noise SNR: Target = {target_snr:.2f} dB, Measured = {measured_snr:.2f} dB")
    assert abs(measured_snr - target_snr) < 0.25, f"SNR deviation {abs(measured_snr - target_snr):.3f} exceeded 0.25 dB!"
    print("  -> Gaussian White Noise SNR test PASSED.")

    # 2. Test Pitch Shifting
    shifted_sig, semitones = pitch_shift_audio(test_sig, sr=16000, min_semitones=-2.0, max_semitones=2.0)
    print(f"[TEST 2] Pitch Shifting: {semitones:+.2f} semitones | Length preserved: {len(shifted_sig)} == {len(test_sig)}")
    assert len(shifted_sig) == len(test_sig), "Pitch shift altered signal length!"
    print("  -> Pitch Shifting test PASSED.")

    # 3. Test SpecAugment (Frequency & Time Masking on Mel Batches)
    spec_batch = torch.ones(16, 1, 128, 128)  # 16 spectrograms
    spec_aug = SpecAugment(freq_mask_param=16, time_mask_param=24, n_freq_masks=2, n_time_masks=2, p=1.0)
    spec_aug.train()
    aug_batch = spec_aug(spec_batch)

    # Verify that values were masked to 0.0
    masked_ratio = float((aug_batch == 0.0).float().mean().item())
    print(f"[TEST 3] SpecAugment Train Batch: Shape = {aug_batch.shape}, Masked Ratio = {masked_ratio * 100:.2f}%")
    assert 0.05 <= masked_ratio <= 0.60, f"Masking ratio {masked_ratio} outside expected range [0.05, 0.60]!"

    # Verify that eval mode completely disables masking
    spec_aug.eval()
    clean_eval_batch = spec_aug(spec_batch)
    eval_masked_ratio = float((clean_eval_batch == 0.0).float().mean().item())
    print(f"[TEST 3] SpecAugment Eval Batch: Masked Ratio = {eval_masked_ratio * 100:.2f}% (Must be 0.0%)")
    assert eval_masked_ratio == 0.0, "SpecAugment masked items during evaluation mode!"
    print("  -> SpecAugment batch masking test PASSED.")

    # 4. Test DataLoader Augmentation Isolation (Train=Active, Val/Test=Clean)
    dummy_X = np.ones((60, 128, 128), dtype=np.float32)
    dummy_y = np.random.randint(0, 7, size=60)

    train_loader, val_loader, test_loader = create_ser_dataloaders(
        dummy_X[:40], dummy_y[:40],
        dummy_X[40:50], dummy_y[40:50],
        dummy_X[50:], dummy_y[50:],
        batch_size=8,
        spec_aug_prob=1.0  # Force augmentation on all training batches for testing
    )

    # Check Train Batch (must contain masked zero stripes)
    train_bx, train_by = next(iter(train_loader))
    train_zeros = float((train_bx == 0.0).float().mean().item())
    print(f"[TEST 4] DataLoader Isolation: Train batch masked = {train_zeros * 100:.2f}%")
    assert train_zeros > 0.0, "Train batch did not receive augmentations!"

    # Check Val & Test Batches (must have 0% masked values)
    val_bx, val_by = next(iter(val_loader))
    val_zeros = float((val_bx == 0.0).float().mean().item())
    test_bx, test_by = next(iter(test_loader))
    test_zeros = float((test_bx == 0.0).float().mean().item())
    print(f"[TEST 4] DataLoader Isolation: Val masked = {val_zeros:.2f}%, Test masked = {test_zeros:.2f}%")
    assert val_zeros == 0.0, "Data leakage/contamination: Val set received augmentations!"
    assert test_zeros == 0.0, "Data leakage/contamination: Test set received augmentations!"
    print("  -> Data augmentation isolation test PASSED.")
    print("=" * 70)
    print("ALL PHASE 2 SELF-TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
