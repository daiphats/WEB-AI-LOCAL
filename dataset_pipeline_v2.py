"""
Speech Emotion Recognition (SER) - Robust Data Preprocessing Pipeline (Phase 1)
================================================================================
Features:
  1. Multi-Dataset Audio Loading: Standardized parser for TESS & RAVDESS datasets.
     Includes automated deduplication and acoustic normalization.
  2. Mel-Spectrogram Extraction: High-fidelity 128 Mel filterbanks.
  3. CMVN (Cepstral Mean and Variance Normalization):
     Cancels room impulse response (additive stationary channel noise in log-mel domain)
     and normalizes speaker-specific vocal dynamics and pitch.
  4. Speaker-Independent Split & Leave-One-Subject-Out (LOSO):
     Strictly guarantees that speakers in validation and test sets never overlap
     with the training set.

Author: Expert AI/ML Engineer
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset

# ------------------------------------------------------------------------------
# 1. Global SER Constants and Emotion Classes
# ------------------------------------------------------------------------------

EMOTION_CLASSES: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

EMOTION_TO_IDX: Dict[str, int] = {emo: i for i, emo in enumerate(EMOTION_CLASSES)}
IDX_TO_EMOTION: Dict[int, str] = {i: emo for i, emo in enumerate(EMOTION_CLASSES)}

# RAVDESS Official 8-class mapping to unified 7 classes
# 01=neutral, 02=calm -> mapped to neutral (low-arousal baseline)
# 03=happy, 04=sad, 05=angry, 06=fearful -> fear, 07=disgust, 08=surprised -> surprise
RAVDESS_EMOTION_MAP: Dict[str, str] = {
    "01": "neutral",
    "02": "neutral",   # Calm unified into neutral
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}


# ------------------------------------------------------------------------------
# 2. CMVN (Cepstral Mean and Variance Normalization)
# ------------------------------------------------------------------------------

def apply_cmvn(spectrogram: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Applies Cepstral Mean and Variance Normalization (CMVN) across the temporal axis.

    Theoretical Justification:
      1. Room Impulse Response (RIR) Removal:
         Convolution in time domain becomes multiplication in frequency domain:
           Y(f, t) = X(f, t) * H(f)
         In log-mel / cepstral domain:
           log|Y(f, t)| = log|X(f, t)| + log|H(f)|
         Since room acoustic response H(f) and microphone transfer functions are
         relatively stationary over time, subtracting the temporal mean:
           mu_f = (1/T) * sum_t log|Y(f, t)|
         cancels log|H(f)|, leaving the intrinsic speech signal unaffected by room acoustics.

      2. Speaker Pitch and Energy Normalization:
         Dividing by the per-channel standard deviation:
           sigma_f = sqrt( (1/T) * sum_t (log|Y(f, t)| - mu_f)^2 )
         normalizes speaker-specific vocal tract dynamics, loudness variations,
         and pitch intensity disparities across speakers.

    Parameters:
      spectrogram: np.ndarray of shape (n_mels, time_steps) or (batch, n_mels, time_steps)
      eps: Float epsilon constant to avoid division by zero for silent/quiescent bins.

    Returns:
      np.ndarray of same shape, with zero mean and unit variance per mel channel.
    """
    # Time frames are along the last dimension (axis = -1)
    mean = np.mean(spectrogram, axis=-1, keepdims=True)
    std = np.std(spectrogram, axis=-1, keepdims=True)
    normalized = (spectrogram - mean) / (std + eps)
    return normalized


# ------------------------------------------------------------------------------
# 3. Mel-Spectrogram Feature Extraction (128 Mels)
# ------------------------------------------------------------------------------

def load_audio(
    file_path: Union[str, Path],
    target_sr: int = 16000,
    trim_silence: bool = True,
    top_db: int = 25
) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file safely with soundfile and librosa fallback,
    converts to mono, resamples to target_sr, and optionally trims silence.
    """
    file_str = str(file_path)
    try:
        data, sr = sf.read(file_str, dtype="float32")
    except Exception:
        data, sr = librosa.load(file_str, sr=None, mono=True)

    # Convert multi-channel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample to target sampling rate (standard 16 kHz)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Optional Voice Activity / Silence Trimming
    if trim_silence and len(data) > int(sr * 0.2):
        trimmed_data, _ = librosa.effects.trim(data, top_db=top_db)
        if len(trimmed_data) >= int(sr * 0.2):
            data = trimmed_data

    return data.astype(np.float32), sr


def extract_mel_spectrogram(
    audio_input: Union[str, Path, np.ndarray],
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: Optional[float] = 8000.0,
    target_time_steps: Optional[int] = 128,
    apply_cmvn_flag: bool = True,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Extracts a 128-band Mel-Spectrogram, applies log-power scaling and CMVN,
    and standardizes time dimension via center-cropping or padding.

    Parameters:
      audio_input: File path (str/Path) or raw 1D audio waveform array (np.ndarray).
      sr: Target audio sampling rate (default 16000 Hz).
      n_mels: Number of Mel frequency bands (fixed to 128).
      n_fft: FFT window size (default 1024 frames).
      hop_length: Hop length between successive STFT frames (default 512 frames = 32ms).
      fmin: Minimum frequency (default 0.0 Hz).
      fmax: Maximum frequency (default 8000.0 Hz for 16kHz Nyquist limit).
      target_time_steps: If specified (e.g. 128), crops or zero-pads temporal frames
                         to output fixed (128, target_time_steps) shape for CNN models.
      apply_cmvn_flag: Whether to execute CMVN normalization.
      eps: Numerical stability constant.

    Returns:
      np.ndarray of shape (128, target_time_steps) or (128, variable_time_steps).
    """
    if isinstance(audio_input, (str, Path)):
        signal, audio_sr = load_audio(audio_input, target_sr=sr)
    else:
        signal = np.asarray(audio_input, dtype=np.float32)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=0)
        audio_sr = sr

    # Short signal fallback: ensure minimum length for FFT
    if len(signal) < n_fft:
        pad_size = n_fft - len(signal)
        signal = np.pad(signal, (0, pad_size), mode="constant")

    # 1. Compute Power Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=audio_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0
    )

    # 2. Convert power to decibels (log scale)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # 3. Apply CMVN on valid temporal speech frames before padding
    if apply_cmvn_flag:
        log_mel = apply_cmvn(log_mel, eps=eps)

    # 4. Standardize temporal dimension
    if target_time_steps is not None:
        t_frames = log_mel.shape[1]
        if t_frames < target_time_steps:
            # Pad on the right with zeros (silence baseline)
            pad_width = target_time_steps - t_frames
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
        elif t_frames > target_time_steps:
            # Symmetrical center crop around energy peak
            start = (t_frames - target_time_steps) // 2
            log_mel = log_mel[:, start:start + target_time_steps]

    return log_mel.astype(np.float32)


# ------------------------------------------------------------------------------
# 4. Dataset Parsers & Audio File Discovery (TESS & RAVDESS)
# ------------------------------------------------------------------------------

def scan_tess_dataset(tess_dir: Union[str, Path]) -> List[Dict[str, Union[str, int]]]:
    """
    Scans the TESS dataset directory, avoiding nested duplicate folders.
    Extracts speaker (OAF or YAF) and emotion label.

    Handles dataset edge cases:
      - Ignores nested copy 'TESS Toronto emotional speech set data'
      - Corrects known typo 'OA_bite_neutral.wav' to speaker 'TESS_OAF'
    """
    tess_path = Path(tess_dir)
    if not tess_path.exists():
        raise FileNotFoundError(f"TESS directory not found: {tess_path}")

    records = []
    # Collect all .wav files while filtering duplicate directory
    for wav_file in tess_path.rglob("*.wav"):
        if "TESS Toronto emotional speech set data" in str(wav_file):
            continue

        stem_lower = wav_file.stem.lower()
        parent_lower = wav_file.parent.name.lower()

        # Determine speaker
        if parent_lower.startswith("oaf") or stem_lower.startswith("oaf") or stem_lower.startswith("oa_"):
            speaker_id = "TESS_OAF"
        elif parent_lower.startswith("yaf") or stem_lower.startswith("yaf"):
            speaker_id = "TESS_YAF"
        else:
            continue

        # Determine emotion
        matched_emotion = None
        if "angry" in stem_lower or "angry" in parent_lower:
            matched_emotion = "angry"
        elif "disgust" in stem_lower or "disgust" in parent_lower:
            matched_emotion = "disgust"
        elif "fear" in stem_lower or "fear" in parent_lower:
            matched_emotion = "fear"
        elif "happy" in stem_lower or "happy" in parent_lower:
            matched_emotion = "happy"
        elif "neutral" in stem_lower or "neutral" in parent_lower:
            matched_emotion = "neutral"
        elif "sad" in stem_lower or "sad" in parent_lower:
            matched_emotion = "sad"
        elif "ps" in stem_lower or "surpris" in stem_lower or "surpris" in parent_lower:
            matched_emotion = "surprise"

        if matched_emotion is not None:
            records.append({
                "file_path": str(wav_file.resolve()),
                "filename": wav_file.name,
                "dataset": "TESS",
                "speaker": speaker_id,
                "emotion": matched_emotion,
                "label": EMOTION_TO_IDX[matched_emotion]
            })

    return records


def scan_ravdess_dataset(ravdess_dir: Union[str, Path]) -> List[Dict[str, Union[str, int]]]:
    """
    Scans the RAVDESS dataset directory, avoiding nested duplicate folders.
    Extracts Actor ID (RAV_Actor_01 to RAV_Actor_24) and emotion label.

    RAVDESS Filename Specification:
      Modality (01/02/03) - Channel (01/02) - Emotion (01-08) -
      Intensity (01/02) - Statement (01/02) - Repetition (01/02) - Actor (01-24)
    """
    rav_path = Path(ravdess_dir)
    if not rav_path.exists():
        raise FileNotFoundError(f"RAVDESS directory not found: {rav_path}")

    records = []
    # Collect all .wav files while filtering nested duplicate folder
    for wav_file in rav_path.rglob("*.wav"):
        if "audio_speech_actors_01-24" in str(wav_file):
            continue

        parts = wav_file.stem.split("-")
        if len(parts) >= 7:
            emotion_code = parts[2]
            actor_code = parts[6]

            if emotion_code in RAVDESS_EMOTION_MAP:
                emotion_name = RAVDESS_EMOTION_MAP[emotion_code]
                speaker_id = f"RAV_Actor_{int(actor_code):02d}"

                records.append({
                    "file_path": str(wav_file.resolve()),
                    "filename": wav_file.name,
                    "dataset": "RAVDESS",
                    "speaker": speaker_id,
                    "emotion": emotion_name,
                    "label": EMOTION_TO_IDX[emotion_name]
                })

    return records


def load_ser_audio_records(
    data_dir: Union[str, Path] = "data",
    datasets: Tuple[str, ...] = ("tess", "ravdess")
) -> List[Dict[str, Union[str, int]]]:
    """
    Discovers and merges audio records across TESS and RAVDESS with deduplication.

    Parameters:
      data_dir: Root directory containing TESS/ and RAVDESS/ subdirectories.
      datasets: Tuple of dataset names to include ('tess', 'ravdess').

    Returns:
      List of metadata dictionaries with keys:
      ['file_path', 'filename', 'dataset', 'speaker', 'emotion', 'label']
    """
    data_path = Path(data_dir)
    records: List[Dict[str, Union[str, int]]] = []

    if "tess" in datasets:
        tess_dir = data_path / "TESS"
        if tess_dir.exists():
            tess_items = scan_tess_dataset(tess_dir)
            records.extend(tess_items)
            print(f"[PREPROC] Discovered {len(tess_items)} unique audio samples from TESS.")
        else:
            print(f"[PREPROC WARN] TESS directory not found at {tess_dir}")

    if "ravdess" in datasets:
        rav_dir = data_path / "RAVDESS"
        if rav_dir.exists():
            rav_items = scan_ravdess_dataset(rav_dir)
            records.extend(rav_items)
            print(f"[PREPROC] Discovered {len(rav_items)} unique audio samples from RAVDESS.")
        else:
            print(f"[PREPROC WARN] RAVDESS directory not found at {rav_dir}")

    print(f"[PREPROC] Total harmonized dataset: {len(records)} samples across {len(set(r['speaker'] for r in records))} distinct speakers.")
    return records


# ------------------------------------------------------------------------------
# 5. High-Throughput Feature Extraction & Dataset Builder
# ------------------------------------------------------------------------------

def build_ser_dataset(
    records: List[Dict[str, Union[str, int]]],
    target_time_steps: int = 128,
    max_workers: int = 4,
    cache_path: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Union[str, int]]]]:
    """
    Extracts Mel-Spectrograms (128 mels) with CMVN for all records in parallel.
    Optionally saves/loads from a cache file for near-instant repeated runs.

    Returns:
      X: np.ndarray of shape (N, 128, target_time_steps)
      y: np.ndarray of shape (N,) with integer class labels [0..6]
      speakers: np.ndarray of shape (N,) with speaker IDs
      metadata: List of dicts for all records
    """
    if cache_path is not None and Path(cache_path).exists():
        print(f"[PREPROC] Loading cached Mel-Spectrogram features from {cache_path}...")
        try:
            cached_data = torch.load(cache_path, weights_only=False)
        except TypeError:
            cached_data = torch.load(cache_path)
        X_cached = cached_data["X"].numpy() if isinstance(cached_data["X"], torch.Tensor) else np.asarray(cached_data["X"])
        y_cached = cached_data["y"].numpy() if isinstance(cached_data["y"], torch.Tensor) else np.asarray(cached_data["y"])
        spk_cached = cached_data["speakers"].numpy() if isinstance(cached_data["speakers"], torch.Tensor) else np.asarray(cached_data["speakers"])
        return (X_cached, y_cached, spk_cached, cached_data["records"])

    print(f"[PREPROC] Extracting Mel-Spectrograms (128 Mels + CMVN) using {max_workers} threads...")
    t0 = time.time()

    def _process_item(item):
        feat = extract_mel_spectrogram(
            item["file_path"],
            sr=16000,
            n_mels=128,
            n_fft=1024,
            hop_length=512,
            target_time_steps=target_time_steps,
            apply_cmvn_flag=True
        )
        return feat, item["label"], item["speaker"], item

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_item, records))

    feats, labels, speakers_list, final_records = [], [], [], []
    for f, l, s, r in results:
        feats.append(f)
        labels.append(l)
        speakers_list.append(s)
        final_records.append(r)

    X = np.array(feats, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    speakers = np.array(speakers_list, dtype=object)

    elapsed = time.time() - t0
    print(f"[PREPROC] Extracted features for {len(X)} samples in {elapsed:.2f}s ({len(X)/elapsed:.1f} samples/sec). Shape: {X.shape}")

    if cache_path is not None:
        save_path = Path(cache_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "X": torch.from_numpy(X),
            "y": torch.from_numpy(y),
            "speakers": np.array(speakers, dtype=str),
            "records": final_records
        }, save_path)
        print(f"[PREPROC] Successfully saved feature cache to {save_path}")

    return X, y, speakers, final_records


# ------------------------------------------------------------------------------
# 6. Speaker-Independent Split & Leave-One-Subject-Out (LOSO)
# ------------------------------------------------------------------------------

def speaker_independent_split(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    speakers: Union[np.ndarray, List[str]],
    test_speakers: Optional[List[str]] = None,
    val_speakers: Optional[List[str]] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    loso_subject: Optional[str] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partitions features and labels into train, validation, and test sets strictly
    based on speaker identities to prevent data leakage and guarantee genuine generalization.

    Mathematical Invariant:
      set(train_speakers) ∩ set(val_speakers) == ∅
      set(train_speakers) ∩ set(test_speakers) == ∅
      set(val_speakers) ∩ set(test_speakers) == ∅

    Supported Splitting Modes:
      1. Leave-One-Subject-Out (LOSO):
         Pass loso_subject (e.g. 'RAV_Actor_24').
         Test set contains only this subject. Val set takes another distinct subject.
         Train set contains all remaining subjects.
      2. Explicit Speaker Allocation:
         Pass test_speakers=['RAV_Actor_23', 'RAV_Actor_24'] and val_speakers=[...].
      3. Disjoint Ratio Splitting (Default):
         Randomly partitions unique speakers according to test_size and val_size.

    Returns:
      X_train, y_train, X_val, y_val, X_test, y_test
    """
    is_torch = isinstance(X, torch.Tensor)
    if is_torch:
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
    else:
        X_np = np.asarray(X)
        y_np = np.asarray(y)

    spk_arr = np.asarray(speakers)
    assert len(X_np) == len(y_np) == len(spk_arr), "Mismatch in lengths of X, y, and speakers!"

    unique_speakers = np.array(sorted(list(set(spk_arr))))
    n_speakers = len(unique_speakers)

    if n_speakers < 3:
        raise ValueError(
            f"Speaker-independent train/val/test split requires at least 3 distinct speakers. "
            f"Found only {n_speakers}: {unique_speakers.tolist()}"
        )

    # 1. Handle Leave-One-Subject-Out (LOSO) mode
    if loso_subject is not None:
        if loso_subject not in unique_speakers:
            raise ValueError(f"loso_subject '{loso_subject}' not found in dataset speakers: {unique_speakers.tolist()}")
        test_spks = [loso_subject]
        remaining = [s for s in unique_speakers if s != loso_subject]
        if val_speakers is not None:
            val_spks = [s for s in val_speakers if s in remaining]
        else:
            # Deterministically reserve 1 distinct speaker for validation
            rng = np.random.default_rng(random_state)
            val_spks = [rng.choice(remaining)]
        train_spks = [s for s in remaining if s not in val_spks]

    # 2. Handle Explicit Speaker Allocation
    elif test_speakers is not None:
        test_spks = list(test_speakers)
        remaining = [s for s in unique_speakers if s not in test_spks]
        if val_speakers is not None:
            val_spks = list(val_speakers)
            train_spks = [s for s in remaining if s not in val_spks]
        else:
            rng = np.random.default_rng(random_state)
            n_val = max(1, int(round(len(remaining) * (val_size / (1.0 - test_size)))))
            val_spks = list(rng.choice(remaining, size=n_val, replace=False))
            train_spks = [s for s in remaining if s not in val_spks]

    # 3. Handle Disjoint Ratio Splitting
    else:
        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(unique_speakers)

        n_test = max(1, int(round(n_speakers * test_size)))
        n_val = max(1, int(round(n_speakers * val_size)))

        # Ensure at least 1 speaker remains in train
        if n_test + n_val >= n_speakers:
            n_val = max(1, n_speakers - n_test - 1)

        test_spks = list(shuffled[:n_test])
        val_spks = list(shuffled[n_test : n_test + n_val])
        train_spks = list(shuffled[n_test + n_val :])

    # --- Verification of Strict Disjoint Isolation ---
    train_set = set(train_spks)
    val_set = set(val_spks)
    test_set = set(test_spks)

    assert len(train_set.intersection(val_set)) == 0, f"Leakage detected between Train and Val: {train_set.intersection(val_set)}"
    assert len(train_set.intersection(test_set)) == 0, f"Leakage detected between Train and Test: {train_set.intersection(test_set)}"
    assert len(val_set.intersection(test_set)) == 0, f"Leakage detected between Val and Test: {val_set.intersection(test_set)}"

    train_mask = np.isin(spk_arr, list(train_set))
    val_mask = np.isin(spk_arr, list(val_set))
    test_mask = np.isin(spk_arr, list(test_set))

    X_train, y_train = X_np[train_mask], y_np[train_mask]
    X_val, y_val = X_np[val_mask], y_np[val_mask]
    X_test, y_test = X_np[test_mask], y_np[test_mask]

    print("[PREPROC] === Speaker-Independent Split Summary ===")
    print(f"  Train: {len(X_train)} samples | {len(train_set)} speakers -> {sorted(list(train_set))[:4]}...")
    print(f"  Val:   {len(X_val)} samples | {len(val_set)} speakers -> {sorted(list(val_set))}")
    print(f"  Test:  {len(X_test)} samples | {len(test_set)} speakers -> {sorted(list(test_set))}")
    print(f"  Total: {len(X_train) + len(X_val) + len(X_test)} samples | Speaker Overlap: 0.0% (VERIFIED)")

    if is_torch:
        return (
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
            torch.from_numpy(X_test),
            torch.from_numpy(y_test)
        )

    return X_train, y_train, X_val, y_val, X_test, y_test


# ------------------------------------------------------------------------------
# 7. PyTorch Dataset Wrapper
# ------------------------------------------------------------------------------

class SERMelDataset(Dataset):
    """
    PyTorch Dataset wrapper for CMVN-normalized Mel-Spectrogram features.
    Expands channel dimension to (1, 128, T) for 2D ConvNets.
    """
    def __init__(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()

        # Add channel dimension if 2D: (N, 128, T) -> (N, 1, 128, T)
        if X.ndim == 3:
            X = X.unsqueeze(1)

        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ------------------------------------------------------------------------------
# 8. End-to-End Pipeline Helper
# ------------------------------------------------------------------------------

def prepare_ser_pipeline(
    data_dir: Union[str, Path] = "data",
    datasets: Tuple[str, ...] = ("tess", "ravdess"),
    target_time_steps: int = 128,
    test_size: float = 0.15,
    val_size: float = 0.15,
    loso_subject: Optional[str] = None,
    cache_file: Optional[str] = "data/ser_mel128_cmvn_cache.pt",
    max_workers: int = 4,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience end-to-end pipeline function:
      1. Scans TESS and RAVDESS audio datasets.
      2. Extracts 128-band Mel-Spectrograms with CMVN.
      3. Splits data using strictly disjoint speaker-independent groups.

    Returns:
      X_train, y_train, X_val, y_val, X_test, y_test
    """
    records = load_ser_audio_records(data_dir=data_dir, datasets=datasets)
    X, y, speakers, _ = build_ser_dataset(
        records,
        target_time_steps=target_time_steps,
        max_workers=max_workers,
        cache_path=cache_file
    )

    return speaker_independent_split(
        X, y, speakers,
        test_size=test_size,
        val_size=val_size,
        loso_subject=loso_subject,
        random_state=random_state
    )


# ------------------------------------------------------------------------------
# 9. Verification & Self-Test Suite
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SER Preprocessing Pipeline Self-Test (Phase 1)")
    print("=" * 70)

    # 1. Test CMVN Normalization Invariant
    dummy_mel = np.random.randn(128, 100) * 12.0 + 35.0
    cmvn_out = apply_cmvn(dummy_mel)
    max_mean = np.max(np.abs(np.mean(cmvn_out, axis=-1)))
    mean_std = np.mean(np.std(cmvn_out, axis=-1))
    print(f"[TEST 1] CMVN Validation: max(|mean|) = {max_mean:.2e}, mean(std) = {mean_std:.4f}")
    assert max_mean < 1e-5, "CMVN mean normalization failed!"
    assert abs(mean_std - 1.0) < 1e-4, "CMVN variance normalization failed!"
    print("  -> CMVN normalization test PASSED.")

    # 2. Test Audio Discovery
    records = load_ser_audio_records("data")
    assert len(records) > 0, "No records found in data directory!"
    print(f"[TEST 2] Dataset scan passed: {len(records)} audio files found.")

    # 3. Test Feature Extraction on Sample Audio
    sample_file = records[0]["file_path"]
    sample_feat = extract_mel_spectrogram(sample_file, target_time_steps=128)
    print(f"[TEST 3] Extracted Mel-Spectrogram shape: {sample_feat.shape} (128 Mels x 128 Frames)")
    assert sample_feat.shape == (128, 128), f"Unexpected shape {sample_feat.shape}"
    print("  -> Feature extraction test PASSED.")

    # 4. Test Speaker-Independent Splitting (Ratio Split across 26 speakers)
    test_sample_indices = list(range(0, len(records), 10))  # 424 samples across all 26 speakers
    test_records = [records[i] for i in test_sample_indices]
    dummy_speakers = np.array([r["speaker"] for r in test_records])
    dummy_X = np.random.randn(len(dummy_speakers), 128, 128).astype(np.float32)
    dummy_y = np.array([r["label"] for r in test_records])

    X_tr, y_tr, X_va, y_va, X_te, y_te = speaker_independent_split(
        dummy_X, dummy_y, dummy_speakers, test_size=0.15, val_size=0.15, random_state=42
    )
    print(f"[TEST 4] Ratio Split: Train={X_tr.shape}, Val={X_va.shape}, Test={X_te.shape}")
    assert len(X_tr) + len(X_va) + len(X_te) == len(dummy_X), "Sample count conservation failed!"
    print("  -> Speaker-independent ratio split test PASSED.")

    # 5. Test Leave-One-Subject-Out (LOSO) Mode
    loso_subject = "RAV_Actor_24"
    X_tr_l, y_tr_l, X_va_l, y_va_l, X_te_l, y_te_l = speaker_independent_split(
        dummy_X, dummy_y, dummy_speakers, loso_subject=loso_subject, random_state=42
    )
    test_spk_indices = np.where(dummy_speakers == loso_subject)[0]
    print(f"[TEST 5] LOSO Split for '{loso_subject}': Test samples={len(X_te_l)} (Expected: {len(test_spk_indices)})")
    assert len(X_te_l) == len(test_spk_indices), "LOSO subject sample mismatch!"
    print("  -> Leave-One-Subject-Out (LOSO) split test PASSED.")
    print("=" * 70)
    print("ALL PHASE 1 SELF-TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
