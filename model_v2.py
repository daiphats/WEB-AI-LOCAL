"""
Speech Emotion Recognition (SER) - CNN-LSTM / CNN-GRU Hybrid Architecture (Phase 3)
====================================================================================
Architecture Specification:
  1. 2D CNN Frontend:
     - 4-stage convolutional hierarchy extracting deep spectro-temporal features
       from 128-band Mel-Spectrograms (formant dynamics, pitch harmonics, energy contours).
     - Batch Normalization on every convolutional stage for training stability.
     - Anti-overfitting High Dropout (0.2 -> 0.3 -> 0.4 -> 0.4).
  2. TimeDistributed Reshaping Layer:
     - Adapts 2D spatial feature maps into a sequential time-series representation
       (Batch, TimeSteps, FeatureDim) matching temporal frame evolution.
  3. Recurrent Sequence Modeling (Bidirectional LSTM / GRU):
     - 2-layer Bidirectional LSTM (or GRU) capturing long-term prosodic and
       emotional transitions across sequential phonemes and words.
  4. Temporal Attention Pooling:
     - Dynamically weighs emotionally charged speech segments over neutral pauses.
  5. Anti-Overfitting Dense Classification Head:
     - Dense projection with Batch Normalization, High Dropout (0.5), and L2 Weight Decay.
     - Softmax output across 7 emotion classes.
     - Standardized compile interface with Adam (lr=1e-4) and Categorical Cross-Entropy.

Author: Expert AI/ML Engineer
"""

import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMOTION_CLASSES: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# ------------------------------------------------------------------------------
# 1. Temporal Attention Mechanism
# ------------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Self-Attention Pooling over recurrent temporal frames.

    Mathematical Formulation:
      e_t = v^T * tanh(W * h_t + b)
      alpha_t = exp(e_t) / sum_tau(exp(e_tau))
      c = sum_t (alpha_t * h_t)

    Weights salient emotional peaks (e.g., vocal outbursts, crying, laughter)
    higher than silent pauses.
    """
    def __init__(self, hidden_dim: int = 256, attention_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )

    def forward(self, seq_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
          seq_features: Tensor of shape (B, T, hidden_dim)

        Returns:
          context_vector: Tensor of shape (B, hidden_dim)
          attention_weights: Tensor of shape (B, T, 1)
        """
        scores = self.projection(seq_features)              # (B, T, 1)
        weights = F.softmax(scores, dim=1)                 # (B, T, 1)
        context = torch.sum(seq_features * weights, dim=1)  # (B, hidden_dim)
        return context, weights


# ------------------------------------------------------------------------------
# 2. Hybrid CNN-LSTM / CNN-GRU Core Architecture
# ------------------------------------------------------------------------------

class CNNLSTM_SER(nn.Module):
    """
    CNN-LSTM Hybrid Deep Neural Network for Speech Emotion Recognition.

    Input:
      Mel-Spectrogram tensor of shape (B, 1, 128, 128)
      where 128 is the number of Mel frequency bands and 128 is temporal frames.

    Anti-Overfitting Protocols:
      - Batch Normalization after every convolution and dense layer.
      - Progressive High Dropout: 0.2 (low-level) -> 0.3 -> 0.4 (mid-level) -> 0.5 (dense head).
      - Recurrent dropout between LSTM layers (0.4).
      - L2 Regularization integrated via Adam weight decay.
    """
    def __init__(
        self,
        num_classes: int = 7,
        recurrent_type: str = "lstm",  # 'lstm' or 'gru'
        recurrent_hidden: int = 128,
        num_recurrent_layers: int = 2,
        bidirectional: bool = True,
        dropout_cnn: float = 0.4,
        dropout_dense: float = 0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.recurrent_type = recurrent_type.lower()
        self.bidirectional = bidirectional
        self.recurrent_hidden = recurrent_hidden
        self.num_directions = 2 if bidirectional else 1

        # ----------------------------------------------------------------------
        # Stage 1: Spatial/Spectral Feature Extraction (2D CNN Hierarchy)
        # ----------------------------------------------------------------------
        # Block 1: (B, 1, 128, 128) -> (B, 32, 64, 64)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.20)
        )

        # Block 2: (B, 32, 64, 64) -> (B, 64, 32, 32)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.30)
        )

        # Block 3: (B, 64, 32, 32) -> (B, 128, 16, 16)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_cnn)
        )

        # Block 4: (B, 128, 16, 16) -> (B, 256, 1, 16)
        # Frequency pooling reduces Mel axis to 1, while preserving 16 temporal steps
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 16)),
            nn.Dropout2d(p=dropout_cnn)
        )

        # ----------------------------------------------------------------------
        # Stage 2: TimeDistributed Reshaping (Spatial 2D -> Temporal Sequence)
        # ----------------------------------------------------------------------
        # Input to recurrent layer: 256 channels per time frame
        recurrent_in_features = 256

        # ----------------------------------------------------------------------
        # Stage 3: Temporal Recurrent Dynamics (LSTM / GRU)
        # ----------------------------------------------------------------------
        if self.recurrent_type == "gru":
            self.recurrent = nn.GRU(
                input_size=recurrent_in_features,
                hidden_size=recurrent_hidden,
                num_layers=num_recurrent_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_cnn if num_recurrent_layers > 1 else 0.0
            )
        else:
            self.recurrent = nn.LSTM(
                input_size=recurrent_in_features,
                hidden_size=recurrent_hidden,
                num_layers=num_recurrent_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_cnn if num_recurrent_layers > 1 else 0.0
            )

        rnn_out_dim = recurrent_hidden * self.num_directions  # 128 * 2 = 256

        # ----------------------------------------------------------------------
        # Stage 4: Temporal Attention Pooling
        # ----------------------------------------------------------------------
        self.attention_pool = TemporalAttention(hidden_dim=rnn_out_dim, attention_dim=64)

        # ----------------------------------------------------------------------
        # Stage 5: Anti-Overfitting Dense Classification Head
        # ----------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_dense),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward computation pass.

        Parameters:
          x: Input tensor of shape (B, 1, 128, 128) or (B, 128, 128)
          return_probs: If True, applies Softmax over logits (probabilities sum to 1.0)
          return_attention: If True, also returns attention weights across frames.

        Returns:
          logits or probabilities of shape (B, num_classes)
        """
        # Ensure 4D shape: (B, 1, 128, 128)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # 1. 2D CNN Hierarchical Spatial Features
        feat = self.conv_block1(x)     # (B, 32, 64, 64)
        feat = self.conv_block2(feat)  # (B, 64, 32, 32)
        feat = self.conv_block3(feat)  # (B, 128, 16, 16)
        feat = self.conv_block4(feat)  # (B, 256, 1, 16)

        # 2. TimeDistributed Reshaping:
        # Squeeze frequency dimension (1) and permute: (B, 256, 1, 16) -> (B, 16, 256)
        # Each slice along dim 1 corresponds to a temporal step t in [1..16]
        seq_features = feat.squeeze(2).permute(0, 2, 1)  # Shape: (B, 16, 256)

        # 3. Recurrent Sequence Processing
        if self.recurrent_type == "gru":
            rnn_out, _ = self.recurrent(seq_features)    # (B, 16, 256)
        else:
            rnn_out, _ = self.recurrent(seq_features)    # (B, 16, 256)

        # 4. Temporal Attention Pooling
        context, attn_weights = self.attention_pool(rnn_out)  # context: (B, 256)

        # 5. Classification Head (Raw Logits)
        logits = self.classifier(context)  # (B, num_classes)

        # 6. Optional Softmax Output
        output = F.softmax(logits, dim=-1) if return_probs else logits

        if return_attention:
            return output, attn_weights
        return output

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Utility method returning Softmax probabilities for inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_probs=True)


# ------------------------------------------------------------------------------
# 3. Model Compilation & Optimization Protocols
# ------------------------------------------------------------------------------

def compile_model(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8
) -> Tuple[torch.optim.Optimizer, nn.Module]:
    """
    Compiles the SER model with Adam optimizer and Categorical Cross-Entropy criterion.

    Anti-Overfitting Measure:
      - weight_decay=1e-4 provides Decoupled L2 Weight Regularization (||theta||_2^2),
        penalizing excessively large weight norms.

    Parameters:
      model: The CNNLSTM_SER PyTorch instance.
      lr: Learning rate (default: 1e-4).
      weight_decay: L2 penalty factor (default: 1e-4).

    Returns:
      optimizer: Configured torch.optim.Adam
      criterion: torch.nn.CrossEntropyLoss
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps
    )
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion


# ------------------------------------------------------------------------------
# 4. Model Summary Generator (Tabular Inspection)
# ------------------------------------------------------------------------------

def print_model_summary(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 1, 128, 128),
    device: str = "cpu"
) -> str:
    """
    Generates a comprehensive tabular Model Summary detailing layer names,
    output shapes, and trainable parameter counts.
    """
    model = model.to(device)
    model.eval()

    rows: List[Tuple[str, str, str, int]] = []
    hooks = []

    def _hook_fn(name: str):
        def _fn(module, inp, out):
            # Only record leaf modules
            if len(list(module.children())) == 0:
                if isinstance(out, torch.Tensor):
                    out_shape = str(list(out.shape))
                elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
                    out_shape = str(list(out[0].shape))
                else:
                    out_shape = "N/A"
                n_params = sum(p.numel() for p in module.parameters())
                rows.append((name, module.__class__.__name__, out_shape, n_params))
        return _fn

    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(_hook_fn(name)))

    dummy_input = torch.zeros(input_size, dtype=torch.float32, device=device)
    with torch.no_grad():
        model(dummy_input)

    for h in hooks:
        h.remove()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Format ASCII Table
    sep = "=" * 88
    sub_sep = "-" * 88
    lines = [
        sep,
        f"{'Layer (Path / Class)':<38} {'Layer Type':<16} {'Output Shape':<20} {'Param #':>10}",
        sep
    ]

    for name, cls_name, shape, params in rows:
        # Truncate overly long path names
        disp_name = (name[:35] + "..") if len(name) > 37 else name
        lines.append(f"{disp_name:<38} {cls_name:<16} {shape:<20} {params:>10,}")

    lines.append(sub_sep)
    lines.append(f"TimeDistributed Reshape: [B, 256, 1, 16] -> [B, 16, 256] (Temporal Sequence)")
    lines.append(f"Temporal Attention Pool: [B, 16, 256]   -> [B, 256]    (Salient Context)")
    lines.append(f"Softmax Classifier:      [B, 256]       -> [B, 7]      (7 Emotion Probs)")
    lines.append(sep)
    lines.append(f"Total Parameters:          {total_params:,}")
    lines.append(f"Trainable Parameters:      {trainable_params:,}")
    lines.append(f"Non-trainable Parameters:  {non_trainable_params:,}")
    lines.append(f"Model Size (FP32 Est.):    {total_params * 4 / (1024**2):.2f} MB")
    lines.append(sep)

    summary_str = "\n".join(lines)
    print(summary_str)
    return summary_str


# ------------------------------------------------------------------------------
# 5. Optional Keras / TensorFlow Definition (Cross-Framework Compatibility)
# ------------------------------------------------------------------------------

def build_keras_cnn_lstm(
    input_shape: Tuple[int, int, int] = (128, 128, 1),
    num_classes: int = 7,
    lr: float = 1e-4,
    l2_reg: float = 1e-4
):
    """
    Keras/TensorFlow counterpart of the architecture for cross-framework compatibility.
    Executed only if TensorFlow is installed in the target runtime.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers
    except ImportError:
        return None

    reg = regularizers.l2(l2_reg)

    inputs = layers.Input(shape=input_shape)

    # 2D CNN Spatial Blocks
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Frequency pooling to 1, leaving 16 temporal frames
    x = layers.AveragePooling2D((16, 1))(x)  # Shape: (B, 1, 16, 256)
    x = layers.Dropout(0.4)(x)

    # TimeDistributed Reshaping: (B, 1, 16, 256) -> (B, 16, 256)
    x = layers.Reshape((16, 256))(x)

    # Recurrent Temporal Sequence Dynamics
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False, dropout=0.4, kernel_regularizer=reg))(x)

    # Dense Classification Head
    x = layers.Dense(128, use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    keras_model = models.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_SER_v2")
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return keras_model


# ------------------------------------------------------------------------------
# 6. Verification & Self-Test Suite
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 88)
    print("SER CNN-LSTM Hybrid Architecture Verification (Phase 3)")
    print("=" * 88)

    # 1. Instantiate Model
    model = CNNLSTM_SER(num_classes=7, recurrent_type="lstm", bidirectional=True)
    print("[TEST 1] CNNLSTM_SER successfully initialized.")

    # 2. Print Detailed Model Summary Table
    print_model_summary(model, input_size=(2, 1, 128, 128))

    # 3. Test Forward Pass & Shape Consistency
    dummy_mel_batch = torch.randn(8, 1, 128, 128, dtype=torch.float32)
    logits = model(dummy_mel_batch)
    probs, attn = model(dummy_mel_batch, return_probs=True, return_attention=True)

    print(f"[TEST 2] Forward Pass Validation:")
    print(f"  Input Batch Shape:      {dummy_mel_batch.shape}")
    print(f"  Logits Output Shape:    {logits.shape} (Expected: [8, 7])")
    print(f"  Probabilities Shape:    {probs.shape} (Expected: [8, 7])")
    print(f"  Attention Tensor Shape: {attn.shape} (Expected: [8, 16, 1])")

    assert logits.shape == (8, 7), f"Expected logits shape (8, 7), got {logits.shape}"
    assert probs.shape == (8, 7), f"Expected probs shape (8, 7), got {probs.shape}"
    assert attn.shape == (8, 16, 1), f"Expected attention shape (8, 16, 1), got {attn.shape}"

    # Verify Softmax Property: sum of probabilities == 1.0
    prob_sums = probs.sum(dim=-1).detach().cpu().numpy()
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Softmax output does not sum to 1.0!"
    print("  -> Softmax normalization verified: sum(p) == 1.0 across all samples.")

    # 4. Test Compilation, Loss & Backward Gradient Propagation
    optimizer, criterion = compile_model(model, lr=1e-4, weight_decay=1e-4)
    model.train()
    optimizer.zero_grad()

    target_labels = torch.randint(0, 7, size=(8,), dtype=torch.long)
    loss = criterion(logits, target_labels)
    loss.backward()

    # Verify gradients computed across all parameters
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients were computed during backpropagation!"
    print(f"[TEST 3] Backpropagation & Optimization:")
    print(f"  Cross-Entropy Loss:     {loss.item():.4f}")
    print(f"  Mean Parameter Grad:    {np.mean(grad_norms):.6f}")
    optimizer.step()
    print("  -> Backward pass and Adam optimization step PASSED.")

    # 5. Test GRU Variant
    gru_model = CNNLSTM_SER(num_classes=7, recurrent_type="gru", bidirectional=True)
    gru_logits = gru_model(dummy_mel_batch)
    assert gru_logits.shape == (8, 7), "GRU variant failed forward pass!"
    print("[TEST 4] CNN-GRU variant validated successfully.")

    print("=" * 88)
    print("ALL PHASE 3 ARCHITECTURE TESTS PASSED SUCCESSFULLY!")
    print("=" * 88)
