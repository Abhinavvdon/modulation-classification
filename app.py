"""Streamlit Interactive Web Application for Project Expo / Live Demo.
Automatic Modulation Classification (AMC) with Real-Time Signal Synthesis,
Constellation Visualization, and Deep Learning / ML Model Inference.
"""

import os
import numpy as np
import torch
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from src.config import ModulationType, SignalConfig, ChannelConfig
from src.signal_generator import generate_single_sample, get_constellation
from src.feature_extraction import extract_features_single, FEATURE_NAMES
from src.deep_models import CLDNN

# Page Configuration for Expo Display
st.set_page_config(
    page_title="AI Modulation Classifier | Project Expo",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished Expo UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #546E7A;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Loads trained PyTorch CLDNN model and classical ML pipelines."""
    models_dir = os.path.join(os.path.dirname(__file__), "results", "saved_models")
    class_names = ["BPSK", "QPSK", "8PSK", "16QAM"]

    # 1. Load PyTorch CLDNN
    dl_model = CLDNN(num_classes=len(class_names), in_channels=2, conv_filters=64, lstm_hidden=64)
    cldnn_path = os.path.join(models_dir, "cldnn_best.pt")
    if os.path.exists(cldnn_path):
        dl_model.load_state_dict(torch.load(cldnn_path, map_location=torch.device("cpu")))
    dl_model.eval()

    # 2. Load Classical ML Pipelines
    baselines = {}
    for name in ["svm", "random_forest", "knn"]:
        p = os.path.join(models_dir, f"{name}_pipeline.joblib")
        if os.path.exists(p):
            baselines[name] = joblib.load(p)

    return dl_model, baselines, class_names


dl_model, baselines, CLASS_NAMES = load_models()

# Header
st.markdown('<p class="main-title">📡 Automatic Modulation Classification (AMC)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-Time Deep Learning (CLDNN) & Machine Learning Signal Demodulation at the Physical Layer</p>', unsafe_allow_html=True)

# Sidebar: Interactive Transmitter & Channel Controls
with st.sidebar:
    st.header("🎛️ Transmitter Controls")
    selected_mod = st.selectbox(
        "Modulation Scheme",
        options=["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"],
        index=1,
        help="Select baseband digital modulation scheme."
    )

    st.subheader("📶 Wireless Channel Impairments")
    snr_val = st.slider(
        "Signal-to-Noise Ratio (SNR in dB)",
        min_value=-10.0,
        max_value=25.0,
        value=10.0,
        step=1.0,
        help="Controls the Additive White Gaussian Noise power."
    )

    channel_type = st.selectbox(
        "Channel Fading Model",
        options=["awgn", "rayleigh", "rician"],
        index=0,
        format_func=lambda x: {"awgn": "Pure AWGN", "rayleigh": "Rayleigh Multipath (NLOS)", "rician": "Rician Fading (LOS)"}[x]
    )

    cfo_val = st.slider(
        "Carrier Frequency Offset (CFO)",
        min_value=0.0,
        max_value=0.03,
        value=0.005,
        step=0.002,
        format="%.3f"
    )

    phase_noise = st.slider(
        "Phase Noise Std Dev (rad)",
        min_value=0.0,
        max_value=0.05,
        value=0.01,
        step=0.005,
        format="%.3f"
    )

    st.markdown("---")
    st.markdown("**🧠 Active AI Model:**")
    model_choice = st.radio(
        "Classifier Architecture",
        options=["CLDNN (Deep Learning)", "SVM (RBF Kernel)", "Random Forest", "KNN (k=5)"],
        index=0
    )

    regen_btn = st.button("⚡ Generate New Frame", use_container_width=True, type="primary")

# Signal Generation
mod_map = {
    "BPSK": ModulationType.BPSK,
    "QPSK": ModulationType.QPSK,
    "8PSK": ModulationType.PSK8,
    "16QAM": ModulationType.QAM16,
    "64QAM": ModulationType.QAM64
}

sig_cfg = SignalConfig(sequence_length=128, samples_per_symbol=8, roll_off=0.35)
chan_cfg = ChannelConfig(
    snr_range_db=[snr_val],
    channel_type=channel_type,
    cfo_max=cfo_val,
    phase_noise_std=phase_noise
)

# Use session state or generate on button
if "current_signal" not in st.session_state or regen_btn:
    rng = np.random.default_rng()
    st.session_state.current_signal = generate_single_sample(
        mod_type=mod_map[selected_mod],
        snr_db=snr_val,
        signal_config=sig_cfg,
        channel_config=chan_cfg,
        rng=rng
    )
    st.session_state.true_mod = selected_mod

iq_frame = st.session_state.current_signal  # Shape: (2, 128)
true_mod = st.session_state.true_mod
i_comp = iq_frame[0]
q_comp = iq_frame[1]
comp_sig = i_comp + 1j * q_comp

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🔬 Live Signal & Constellation",
    "🤖 Real-Time AI Inference",
    "📊 Benchmarks & Expo Diagnostics"
])

# -------------------------------------------------------------
# TAB 1: Live Signal & Constellation
# -------------------------------------------------------------
with tab1:
    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        st.subheader("🌌 Constellation Diagram (I vs. Q)")
        fig_const, ax_c = plt.subplots(figsize=(5, 5))
        # Plot ideal constellation
        ideal_c = get_constellation(mod_map[true_mod])
        ax_c.scatter(np.real(ideal_c), np.imag(ideal_c), color="#D32F2F", s=90, marker="+", label="Ideal Constellation", zorder=3)
        # Plot received noisy symbols
        ax_c.scatter(i_comp, q_comp, color="#1976D2", alpha=0.65, s=30, label="Received Signal", zorder=2)
        ax_c.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_c.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax_c.set_xlim(-2.5, 2.5)
        ax_c.set_ylim(-2.5, 2.5)
        ax_c.set_xlabel("In-Phase (I)", fontweight="bold")
        ax_c.set_ylabel("Quadrature (Q)", fontweight="bold")
        ax_c.grid(True, linestyle=":", alpha=0.6)
        ax_c.legend(loc="upper right", fontsize=8)
        st.pyplot(fig_const, use_container_width=True)
        plt.close(fig_const)

    with col_c2:
        st.subheader("📈 Power Spectral Density (PSD)")
        fig_psd, ax_p = plt.subplots(figsize=(5, 5))
        psd = np.abs(np.fft.fftshift(np.fft.fft(comp_sig))) ** 2
        psd_db = 10 * np.log10(psd + 1e-12)
        freqs = np.linspace(-4, 4, len(psd))  # Normalized MHz
        ax_p.plot(freqs, psd_db, color="#00897B", linewidth=1.5)
        ax_p.set_xlabel("Frequency (MHz)", fontweight="bold")
        ax_p.set_ylabel("Relative Power (dB)", fontweight="bold")
        ax_p.set_title(f"Spectrum (SNR: {snr_val:.1f} dB)", fontsize=10)
        ax_p.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig_psd, use_container_width=True)
        plt.close(fig_psd)

    st.subheader("🌊 Baseband I & Q Time-Domain Waveforms")
    fig_time, ax_t = plt.subplots(figsize=(12, 2.8))
    t_samples = np.arange(len(i_comp))
    ax_t.plot(t_samples, i_comp, label="In-Phase (I)", color="#1976D2", linewidth=1.8)
    ax_t.plot(t_samples, q_comp, label="Quadrature (Q)", color="#E64A19", linewidth=1.8, linestyle="--")
    ax_t.set_xlabel("Sample Index n", fontweight="bold")
    ax_t.set_ylabel("Amplitude", fontweight="bold")
    ax_t.grid(True, linestyle=":", alpha=0.6)
    ax_t.legend(loc="upper right")
    st.pyplot(fig_time, use_container_width=True)
    plt.close(fig_time)

# -------------------------------------------------------------
# TAB 2: Real-Time AI Inference
# -------------------------------------------------------------
with tab2:
    st.subheader("🎯 Real-Time AMC Model Prediction")

    # Run Inference
    feats = extract_features_single(iq_frame)

    if "CLDNN" in model_choice:
        with torch.no_grad():
            tensor_in = torch.from_numpy(iq_frame).unsqueeze(0).float()
            logits = dl_model(tensor_in)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100.0
    else:
        pipeline_key = {
            "SVM (RBF Kernel)": "svm",
            "Random Forest": "random_forest",
            "KNN (k=5)": "knn"
        }[model_choice]

        if pipeline_key in baselines:
            pipe = baselines[pipeline_key]
            feats_2d = feats.reshape(1, -1)
            pred_idx = int(pipe.predict(feats_2d)[0])
            pred_class = CLASS_NAMES[pred_idx]
            if hasattr(pipe, "predict_proba"):
                probs = pipe.predict_proba(feats_2d)[0]
                confidence = probs[pred_idx] * 100.0
            else:
                probs = np.zeros(len(CLASS_NAMES))
                probs[pred_idx] = 1.0
                confidence = 100.0
        else:
            pred_class = "Model not found"
            probs = np.zeros(len(CLASS_NAMES))
            confidence = 0.0

    # Display Verdict
    col_v1, col_v2, col_v3 = st.columns([1, 1, 1])
    with col_v1:
        st.metric("Transmitted (True) Modulation", true_mod)
    with col_v2:
        is_correct = (pred_class == true_mod)
        st.metric(
            "AI Predicted Modulation",
            pred_class,
            delta="CORRECT" if is_correct else "MISCLASSIFIED",
            delta_color="normal" if is_correct else "inverse"
        )
    with col_v3:
        st.metric("Model Confidence", f"{confidence:.1f}%")

    st.markdown("---")
    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        st.markdown("**Softmax Class Probability Distribution:**")
        fig_prob, ax_pb = plt.subplots(figsize=(6, 3.5))
        bars = ax_pb.bar(CLASS_NAMES, probs * 100.0, color="#1E88E5", edgecolor="#0D47A1", alpha=0.85)
        # Highlight top prediction
        if pred_idx < len(bars):
            bars[pred_idx].set_color("#43A047" if is_correct else "#E53935")
        ax_pb.set_ylabel("Probability (%)", fontweight="bold")
        ax_pb.set_ylim(0, 105)
        ax_pb.grid(axis="y", linestyle=":", alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax_pb.text(bar.get_x() + bar.get_width() / 2., height + 2, f"{height:.1f}%", ha="center", va="bottom", fontsize=9)
        st.pyplot(fig_prob, use_container_width=True)
        plt.close(fig_prob)

    with col_p2:
        st.markdown("**Extracted Domain Features (HOCs & Spectral Moments):**")
        feature_dict = {
            "Normalized C40 (|C40|/C21^2)": f"{feats[6]:.4f}",
            "Normalized C42 (|C42|/C21^2)": f"{feats[7]:.4f}",
            "Normalized C63 (|C63|/C21^3)": f"{feats[8]:.4f}",
            "Max PSD Instantaneous Amp (gamma_max)": f"{feats[9]:.4f}",
            "Std Dev Phase Non-Linear (sigma_dp)": f"{feats[11]:.4f}",
            "Envelope Skewness": f"{feats[16]:.4f}",
            "Envelope Kurtosis": f"{feats[17]:.4f}",
            "I/Q Cross-Correlation": f"{feats[24]:.4f}",
        }
        st.table(feature_dict)

# -------------------------------------------------------------
# TAB 3: Benchmarks & Expo Diagnostics
# -------------------------------------------------------------
with tab3:
    st.subheader("📊 Systematic Evaluation Across SNR Conditions")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    curve_img = os.path.join(results_dir, "accuracy_vs_snr_comparison.png")
    cldnn_cm = os.path.join(results_dir, "cldnn_confusion_matrix.png")
    svm_cm = os.path.join(results_dir, "svm_confusion_matrix.png")

    if os.path.exists(curve_img):
        st.image(curve_img, caption="Accuracy vs. SNR (dB) across SVM, Random Forest, KNN, and CLDNN", use_container_width=True)

    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        if os.path.exists(cldnn_cm):
            st.image(cldnn_cm, caption="Primary Model: CLDNN Confusion Matrix", use_container_width=True)
    with col_cm2:
        if os.path.exists(svm_cm):
            st.image(svm_cm, caption="Baseline Model: SVM Confusion Matrix", use_container_width=True)

    st.markdown("""
    ### 🏆 Why Deep Learning (CLDNN) Beats Classical ML in Wireless AMC:
    1. **End-to-End Representation**: CLDNN accepts raw I/Q time-series tensors `(2, 128)` directly without lossy manual feature compression.
    2. **Multi-Scale Spatial-Temporal Learning**: 1D Convolutions learn short-term pulse-shaping transitions, while Bidirectional LSTMs retain long-term inter-symbol memory.
    3. **Robustness Under Extreme Noise**: Even at $-6$ dB SNR, the temporal attention mechanism dynamically weights high-confidence symbol transitions.
    """)
