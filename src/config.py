"""Configuration module for Automatic Modulation Classification (AMC).
Defines dataclasses and constants for signal generation, channel impairments,
feature extraction, and model training.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional


class ModulationType(str, Enum):
    BPSK = "BPSK"
    QPSK = "QPSK"
    PSK8 = "8PSK"
    QAM16 = "16QAM"
    QAM64 = "64QAM"

    @classmethod
    def default_classes(cls) -> List["ModulationType"]:
        """Returns the core 4 modulations specified: BPSK, QPSK, 8PSK, 16QAM."""
        return [cls.BPSK, cls.QPSK, cls.PSK8, cls.QAM16]

    @classmethod
    def all_classes(cls) -> List["ModulationType"]:
        """Returns all supported modulations."""
        return [cls.BPSK, cls.QPSK, cls.PSK8, cls.QAM16, cls.QAM64]


@dataclass
class ChannelConfig:
    """Configures wireless channel impairments and noise conditions."""
    # SNR in dB range
    snr_range_db: List[float] = field(
        default_factory=lambda: [-10.0, -6.0, -2.0, 0.0, 2.0, 6.0, 10.0, 14.0, 18.0, 20.0]
    )
    # Channel type: 'awgn', 'rayleigh', 'rician'
    channel_type: str = "awgn"
    # Rician K-factor (ratio of LOS to diffuse power in linear scale)
    rician_k: float = 3.0
    # Multipath tap delays and powers (dB)
    multipath_delays: Tuple[int, ...] = (0, 1, 2)
    multipath_powers_db: Tuple[float, ...] = (0.0, -3.0, -8.0)
    # Carrier frequency offset (normalized to sampling frequency) max range [-cfo_max, +cfo_max]
    cfo_max: float = 0.01
    # Phase noise standard deviation in radians
    phase_noise_std: float = 0.02
    # Apply fractional timing offset
    apply_timing_jitter: bool = False


@dataclass
class SignalConfig:
    """Configures baseband modulation and pulse shaping parameters."""
    samples_per_symbol: int = 8
    symbol_rate: float = 1e6
    sample_rate: float = 8e6
    roll_off: float = 0.35  # RRC filter roll-off factor (alpha)
    filter_span_symbols: int = 8
    # Window length in I/Q samples (e.g. 128 as in RadioML benchmarks)
    sequence_length: int = 128
    # Number of samples generated per (modulation, SNR) combination for dataset creation
    samples_per_snr_per_mod: int = 200
    random_seed: int = 42


@dataclass
class ExperimentConfig:
    """Top-level pipeline configuration."""
    signal: SignalConfig = field(default_factory=SignalConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    modulations: List[ModulationType] = field(default_factory=ModulationType.default_classes)
    # Train / Val / Test split ratios
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    # Model parameters
    batch_size: int = 64
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    random_seed: int = 42
