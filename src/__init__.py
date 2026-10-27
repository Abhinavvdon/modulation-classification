"""Automatic Modulation Classification (AMC) Core Module.
Contains signal generation, feature extraction, baseline ML models, and deep learning architectures.
"""

from .config import ModulationType, SignalConfig, ChannelConfig, ExperimentConfig
from .signal_generator import generate_dataset, generate_single_sample, get_constellation
from .feature_extraction import extract_features_dataset, extract_features_single, FEATURE_NAMES
from .baseline_models import BaselineModelSuite
from .deep_models import CLDNN, AMCResNet1D, get_deep_model
from .dataset import ModulationDataset, split_dataset, create_dataloaders
from .train_eval import train_deep_model, evaluate_deep_model, plot_accuracy_vs_snr, plot_confusion_matrix

__all__ = [
    "ModulationType",
    "SignalConfig",
    "ChannelConfig",
    "ExperimentConfig",
    "generate_dataset",
    "generate_single_sample",
    "get_constellation",
    "extract_features_dataset",
    "extract_features_single",
    "FEATURE_NAMES",
    "BaselineModelSuite",
    "CLDNN",
    "AMCResNet1D",
    "get_deep_model",
    "ModulationDataset",
    "split_dataset",
    "create_dataloaders",
    "train_deep_model",
    "evaluate_deep_model",
    "plot_accuracy_vs_snr",
    "plot_confusion_matrix",
]
