"""Training and Evaluation routines for Deep Learning and Baseline Models.

Provides:
- PyTorch training loop with CosineAnnealingLR and Early Stopping
- Deep Learning evaluation across SNR conditions
- Multi-model benchmarking (SVM, Random Forest, KNN, CLDNN)
- Visualization utilities:
  - Accuracy vs. SNR curves
  - Confusion matrices (Overall & per-SNR)
  - Tabular performance reporting
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt


def get_device() -> torch.device:
    """Detects and returns optimal available hardware accelerator (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_deep_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    early_stop_patience: int = 7,
    save_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """Trains a PyTorch modulation classification model with early stopping.
    
    Returns:
        History dictionary with 'train_loss', 'val_loss', 'val_acc'.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        running_train_loss = 0.0
        total_train_samples = 0

        for signals, labels, _ in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for recurrent / attention stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            running_train_loss += loss.item() * signals.size(0)
            total_train_samples += signals.size(0)

        scheduler.step()
        epoch_train_loss = running_train_loss / max(total_train_samples, 1)

        # 2. Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val_samples = 0

        with torch.no_grad():
            for signals, labels, _ in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)

                outputs = model(signals)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * signals.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val_samples += signals.size(0)

        epoch_val_loss = running_val_loss / max(total_val_samples, 1)
        epoch_val_acc = correct_val / max(total_val_samples, 1)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # Early stopping and checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    # Reload best checkpoint if saved
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return history


def evaluate_deep_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Evaluates trained PyTorch model overall and per SNR."""
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_snrs: List[float] = []

    with torch.no_grad():
        for signals, labels, snrs in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())
            all_snrs.extend(snrs.numpy().tolist())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    snrs_arr = np.array(all_snrs)

    overall_acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Per-SNR breakdown
    unique_snrs = np.sort(np.unique(snrs_arr))
    snr_accuracies = {}
    for snr in unique_snrs:
        mask = (snrs_arr == snr)
        if np.any(mask):
            acc = float(accuracy_score(y_true[mask], y_pred[mask]))
            snr_accuracies[float(snr)] = acc

    return {
        "overall_accuracy": overall_acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "snr_accuracies": snr_accuracies,
        "predictions": y_pred,
        "targets": y_true,
        "snrs": snrs_arr,
    }


def plot_accuracy_vs_snr(
    results_by_model: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "Modulation Classification: Accuracy vs. SNR"
) -> None:
    """Plots and saves multi-model Accuracy vs SNR comparison curve."""
    plt.figure(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "v", "p"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (model_name, snr_dict) in enumerate(results_by_model.items()):
        snrs = sorted(snr_dict.keys())
        accs = [snr_dict[s] * 100.0 for s in snrs]
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        plt.plot(
            snrs,
            accs,
            marker=marker,
            label=model_name,
            linewidth=2,
            markersize=7,
            color=color
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Signal-to-Noise Ratio (SNR in dB)", fontsize=12)
    plt.ylabel("Classification Accuracy (%)", fontsize=12)
    plt.ylim(-2, 102)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix"
) -> None:
    """Plots and saves normalized confusion matrix heatmap."""
    cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.colorbar(label="Normalized Ratio")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    # Annotate numbers
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
            color = "white" if cm_norm[i, j] > thresh else "black"
            plt.text(j, i, val_str, horizontalalignment="center", verticalalignment="center", color=color, fontsize=9)

    plt.ylabel("True Modulation Class", fontsize=12)
    plt.xlabel("Predicted Modulation Class", fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
