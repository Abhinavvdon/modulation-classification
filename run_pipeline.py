"""End-to-End Execution Script for Modulation Classification (Milestone 2).

Runs the complete workflow:
1. Signal Generation (I/Q datasets for BPSK, QPSK, 8PSK, 16QAM under diverse SNR & channel conditions)
2. Feature Extraction (HOCs, Azzouz-Nandi spectral/phase moments)
3. Baseline Models Training & Evaluation (SVM, Random Forest, KNN)
4. Primary Deep Learning Model Training & Evaluation (CLDNN on raw I/Q)
5. Comprehensive Performance Comparison (SNR curves, confusion matrices, metrics)
"""

import os
import time
import argparse
import numpy as np

from src.config import (
    ModulationType,
    SignalConfig,
    ChannelConfig,
    ExperimentConfig
)
from src.signal_generator import generate_dataset
from src.feature_extraction import extract_features_dataset, FEATURE_NAMES
from src.baseline_models import BaselineModelSuite
from src.deep_models import CLDNN
from src.dataset import split_dataset, create_dataloaders, save_dataset_npz
from src.train_eval import (
    get_device,
    train_deep_model,
    evaluate_deep_model,
    plot_accuracy_vs_snr,
    plot_confusion_matrix
)


def parse_args():
    parser = argparse.ArgumentParser(description="AMC Core Module Prototyping Pipeline")
    parser.add_argument("--samples-per-snr", type=int, default=150, help="Samples per modulation per SNR level")
    parser.add_argument("--sequence-length", type=int, default=128, help="I/Q window length in samples")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for Deep Learning training")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save plots and artifacts")
    parser.add_argument("--channel-type", type=str, default="awgn", choices=["awgn", "rayleigh", "rician"])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print(" MILENSTONE 2: AUTOMATIC MODULATION CLASSIFICATION (AMC) PIPELINE ")
    print("=" * 75)

    device = get_device()
    print(f"[*] Compute Device: {device} ({'GPU Acceleration Active' if device.type == 'cuda' else 'CPU Mode'})")

    # ---------------------------------------------------------
    # 1. Configuration & Signal Generation
    # ---------------------------------------------------------
    snr_levels = [-6.0, -2.0, 0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    exp_cfg = ExperimentConfig(
        signal=SignalConfig(
            samples_per_symbol=8,
            sequence_length=args.sequence_length,
            samples_per_snr_per_mod=args.samples_per_snr,
            roll_off=0.35,
            random_seed=42
        ),
        channel=ChannelConfig(
            snr_range_db=snr_levels,
            channel_type=args.channel_type,
            cfo_max=0.005,
            phase_noise_std=0.01
        ),
        modulations=ModulationType.default_classes(),
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    print(f"\n[1/5] Generating Synthetic I/Q Dataset...")
    print(f"      Modulations: {[m.value for m in exp_cfg.modulations]}")
    print(f"      SNR range: {snr_levels} dB (Channel: {args.channel_type.upper()})")
    print(f"      Window length: {args.sequence_length} I/Q samples")
    
    t0 = time.time()
    signals, labels, snrs, class_names = generate_dataset(exp_cfg)
    total_samples = len(labels)
    print(f"      Generated {total_samples} I/Q samples in {time.time() - t0:.2f}s.")
    print(f"      Signals shape: {signals.shape}, Labels shape: {labels.shape}")

    # Save generated raw dataset
    raw_dataset_path = os.path.join(args.output_dir, "amc_dataset.npz")
    save_dataset_npz(raw_dataset_path, signals, labels, snrs, class_names)
    print(f"      Saved dataset archive to: {raw_dataset_path}")

    # ---------------------------------------------------------
    # 2. Dataset Splitting
    # ---------------------------------------------------------
    print(f"\n[2/5] Performing Stratified Train / Val / Test Split (70% / 15% / 15%)...")
    splits = split_dataset(
        signals=signals,
        labels=labels,
        snrs=snrs,
        train_ratio=exp_cfg.train_split,
        val_ratio=exp_cfg.val_split,
        test_ratio=exp_cfg.test_split,
        random_seed=42
    )

    X_train_raw, y_train, snr_train = splits["train"]
    X_val_raw, y_val, snr_val = splits["val"]
    X_test_raw, y_test, snr_test = splits["test"]

    print(f"      Train samples: {len(y_train)} | Val samples: {len(y_val)} | Test samples: {len(y_test)}")

    # ---------------------------------------------------------
    # 3. Feature Extraction for Classical ML Baselines
    # ---------------------------------------------------------
    print(f"\n[3/5] Extracting Hand-Crafted Features for Classical ML Baselines...")
    print(f"      Extracting {len(FEATURE_NAMES)} features (Higher-Order Cumulants + Azzouz-Nandi moments)...")
    t0 = time.time()
    X_train_feats = extract_features_dataset(X_train_raw)
    X_test_feats = extract_features_dataset(X_test_raw)
    print(f"      Feature extraction finished in {time.time() - t0:.2f}s.")
    print(f"      Train feature matrix: {X_train_feats.shape}, Test feature matrix: {X_test_feats.shape}")

    # ---------------------------------------------------------
    # 4. Train & Evaluate Classical ML Baselines (SVM, RF, KNN)
    # ---------------------------------------------------------
    print(f"\n[4/5] Training Classical ML Baselines (SVM, Random Forest, KNN)...")
    suite = BaselineModelSuite(random_state=42)

    for model_name in ["SVM", "Random_Forest", "KNN"]:
        t_start = time.time()
        suite.train(model_name, X_train_feats, y_train)
        print(f"      [+] Trained {model_name:<14} in {time.time() - t_start:.2f}s")

    # Evaluate ML baselines
    baseline_eval = suite.evaluate_all(
        X_test_feats,
        y_test,
        snrs_test=snr_test,
        class_names=class_names
    )

    for name, res in baseline_eval.items():
        print(f"      -> {name:<14} Overall Test Accuracy: {res['overall_accuracy'] * 100.0:.2f}%")

    # Save trained baseline models
    models_dir = os.path.join(args.output_dir, "saved_models")
    suite.save(models_dir)

    # ---------------------------------------------------------
    # 5. Train & Evaluate Primary Deep Learning Model (CLDNN)
    # ---------------------------------------------------------
    print(f"\n[5/5] Training Primary Deep Learning Model (CLDNN with Temporal Attention)...")
    dataloaders = create_dataloaders(splits, batch_size=exp_cfg.batch_size)
    cldnn_model = CLDNN(
        num_classes=len(class_names),
        in_channels=2,
        conv_filters=64,
        lstm_hidden=64,
        dropout=0.3
    )
    checkpoint_path = os.path.join(models_dir, "cldnn_best.pt")

    t_dl_start = time.time()
    history = train_deep_model(
        model=cldnn_model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        epochs=exp_cfg.epochs,
        lr=exp_cfg.learning_rate,
        device=device,
        save_path=checkpoint_path
    )
    print(f"      CLDNN trained in {time.time() - t_dl_start:.2f}s ({len(history['train_loss'])} epochs executed).")

    # Evaluate Deep Learning model
    dl_eval = evaluate_deep_model(
        cldnn_model,
        dataloaders["test"],
        class_names=class_names,
        device=device
    )
    print(f"      -> CLDNN (Deep Learning) Overall Test Accuracy: {dl_eval['overall_accuracy'] * 100.0:.2f}%")

    # ---------------------------------------------------------
    # 6. Results Compilation, Visualizations, and Reporting
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print(" SUMMARY PERFORMANCE COMPARISON (OVERALL ACCURACY)")
    print("=" * 75)
    all_models_eval = {**baseline_eval, "CLDNN (Deep Learning)": dl_eval}

    print(f"{'Model Architecture':<25} | {'Overall Test Acc (%)':<22} | {'Input Representation'}")
    print("-" * 75)
    for model_name, res in all_models_eval.items():
        input_type = "Raw I/Q Tensors (2, 128)" if "CLDNN" in model_name else "25-D Extracted Features (HOCs + Moments)"
        print(f"{model_name:<25} | {res['overall_accuracy'] * 100.0:>20.2f}% | {input_type}")

    # Accuracy vs SNR Comparison
    snr_curve_dict = {}
    for name, res in all_models_eval.items():
        snr_curve_dict[name] = res["snr_accuracies"]

    curve_plot_path = os.path.join(args.output_dir, "accuracy_vs_snr_comparison.png")
    plot_accuracy_vs_snr(snr_curve_dict, curve_plot_path, title=f"AMC Model Comparison (Channel: {args.channel_type.upper()})")
    print(f"\n[+] Saved Accuracy vs. SNR plot to: {curve_plot_path}")

    # Confusion Matrices for CLDNN and SVM
    cldnn_cm_path = os.path.join(args.output_dir, "cldnn_confusion_matrix.png")
    plot_confusion_matrix(
        np.array(dl_eval["confusion_matrix"]),
        class_names,
        cldnn_cm_path,
        title="CLDNN Deep Learning - Test Confusion Matrix"
    )
    print(f"[+] Saved CLDNN Confusion Matrix to: {cldnn_cm_path}")

    svm_cm_path = os.path.join(args.output_dir, "svm_confusion_matrix.png")
    plot_confusion_matrix(
        np.array(baseline_eval["SVM"]["confusion_matrix"]),
        class_names,
        svm_cm_path,
        title="SVM Baseline - Test Confusion Matrix"
    )
    print(f"[+] Saved SVM Confusion Matrix to: {svm_cm_path}")

    # Print Detailed Per-SNR Breakdown Table
    print("\n" + "=" * 75)
    print(" ACCURACY (%) BREAKDOWN BY SNR (dB)")
    print("=" * 75)
    header = f"{'SNR (dB)':<10}" + "".join([f"| {name:<14}" for name in all_models_eval.keys()])
    print(header)
    print("-" * len(header))
    for snr in sorted(snr_levels):
        row = f"{snr:<10.1f}"
        for name in all_models_eval.keys():
            acc = all_models_eval[name]["snr_accuracies"].get(float(snr), 0.0) * 100.0
            row += f"| {acc:>12.1f}% "
        print(row)

    print("\n[+] Milestone 2 Prototyping Pipeline completed successfully!")


if __name__ == "__main__":
    main()
