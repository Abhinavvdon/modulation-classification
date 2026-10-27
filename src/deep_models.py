"""Primary Deep Learning Architectures for Automatic Modulation Classification (AMC).

Designed specifically for raw baseband I/Q signal tensors of shape:
(Batch_Size, 2, Sequence_Length) where channel 0 is In-phase (I) and channel 1 is Quadrature (Q).

Architectures:
1. CLDNN (Convolutional LSTM Deep Neural Network with Attention):
   - 1D Conv Blocks for multi-scale local spatial/phase pattern extraction
   - Bidirectional LSTM for long-range temporal symbol transitions
   - Temporal Attention Mechanism for weighted sequence aggregation
   - Fully Connected Dense Head
2. AMCResNet1D (Residual 1D Network for RF signals):
   - 1D Residual blocks with shortcut connections, BatchNorm, and Dropout
   - Global Average Pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalAttention(nn.Module):
    """Self-attention mechanism over sequence time-steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_outputs: torch.Tensor) -> torch.Tensor:
        """Args:

        rnn_outputs: (Batch, Seq_Len, Hidden_Dim)
        Returns:
            context_vector: (Batch, Hidden_Dim)
        """
        # energy: (Batch, Seq_Len, 1)
        energy = self.projection(rnn_outputs)
        weights = F.softmax(energy, dim=1)
        # context: (Batch, Hidden_Dim)
        context = torch.sum(weights * rnn_outputs, dim=1)
        return context


class CLDNN(nn.Module):
    """Primary Deep Learning Model: Convolutional LSTM Deep Neural Network with Attention.
    
    Accepts raw I/Q frames: (Batch, 2, Sequence_Length).
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 2,
        conv_filters: int = 64,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        # Conv Block 1: Large receptive field to capture initial pulse shape
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=conv_filters,
            kernel_size=7,
            padding=3
        )
        self.bn1 = nn.BatchNorm1d(conv_filters)

        # Conv Block 2: Medium receptive field
        self.conv2 = nn.Conv1d(
            in_channels=conv_filters,
            out_channels=conv_filters * 2,
            kernel_size=5,
            padding=2
        )
        self.bn2 = nn.BatchNorm1d(conv_filters * 2)

        # Conv Block 3: Fine feature extraction
        self.conv3 = nn.Conv1d(
            in_channels=conv_filters * 2,
            out_channels=conv_filters * 2,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm1d(conv_filters * 2)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)

        # Bidirectional LSTM: captures forward/backward symbol memory
        # Input to LSTM needs (Batch, Seq_Len, Features)
        conv_out_dim = conv_filters * 2
        self.lstm = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Attention layer over 2 * lstm_hidden (due to bidirectional)
        self.attention = TemporalAttention(hidden_dim=lstm_hidden * 2)

        # Dense classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: (Batch, 2, Sequence_Length)
        Returns:
            logits: (Batch, num_classes)
        """
        # CNN layers
        h = F.gelu(self.bn1(self.conv1(x)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = self.pool(h)
        h = F.gelu(self.bn3(self.conv3(h)))
        h = self.dropout(h)

        # Prepare for LSTM: transpose from (Batch, Channels, Time) to (Batch, Time, Channels)
        h = h.permute(0, 2, 1)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(h)  # (Batch, Time, 2 * lstm_hidden)

        # Temporal Attention pooling
        attended = self.attention(lstm_out)  # (Batch, 2 * lstm_hidden)

        # Classification logits
        logits = self.classifier(attended)
        return logits


class ResidualBlock1D(nn.Module):
    """1D Residual Block with BatchNorm and Dropout."""

    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = F.gelu(out + residual)
        return out


class AMCResNet1D(nn.Module):
    """Deep Residual Network baseline for RF I/Q Classification."""

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 2,
        base_channels: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.GELU()
        )

        self.res1 = ResidualBlock1D(base_channels, dropout=dropout)
        self.down1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.GELU()
        )
        self.res2 = ResidualBlock1D(base_channels * 2, dropout=dropout)
        self.down2 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.GELU()
        )
        self.res3 = ResidualBlock1D(base_channels * 4, dropout=dropout)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.entry(x)
        h = self.res1(h)
        h = self.down1(h)
        h = self.res2(h)
        h = self.down2(h)
        h = self.res3(h)
        h = self.gap(h).squeeze(-1)
        logits = self.classifier(h)
        return logits


def get_deep_model(
    model_name: str = "CLDNN",
    num_classes: int = 4,
    in_channels: int = 2,
    dropout: float = 0.3
) -> nn.Module:
    """Factory function for instantiating deep learning models."""
    name_upper = model_name.upper()
    if name_upper in ["CLDNN", "PRIMARY"]:
        return CLDNN(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    elif name_upper in ["RESNET", "AMC_RESNET", "RESNET1D"]:
        return AMCResNet1D(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    else:
        raise ValueError(f"Unknown deep learning model architecture: {model_name}")
