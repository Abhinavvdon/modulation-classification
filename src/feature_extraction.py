"""Feature Extraction Pipeline for Automatic Modulation Classification (AMC).

Extracts domain-expert hand-crafted features from raw I/Q signal frames:
1. Higher-Order Statistics & Cumulants (HOCs):
   - Second-order cumulants: C20, C21
   - Fourth-order cumulants: C40, C41, C42
   - Sixth-order cumulants: C60, C63
   - Normalized cumulant ratios (invariant to signal power/scaling)
2. Instantaneous Spectral & Temporal Moments (Azzouz & Nandi features):
   - gamma_max: Maximum value of the power spectral density of normalized centered instantaneous amplitude
   - sigma_ap: Standard deviation of absolute non-linear instantaneous phase
   - sigma_dp: Standard deviation of direct non-linear instantaneous phase
   - sigma_aa: Standard deviation of centered instantaneous amplitude
   - sigma_af: Standard deviation of centered instantaneous frequency
3. Statistical Distribution Moments:
   - Variance, Skewness, Kurtosis of I, Q, and Envelope
   - I/Q Cross-Correlation coefficient
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Dict


FEATURE_NAMES = [
    # Higher-Order Cumulants (magnitudes & ratios)
    "abs_C20",
    "abs_C21",
    "abs_C40",
    "abs_C41",
    "abs_C42",
    "abs_C63",
    "norm_C40",
    "norm_C42",
    "norm_C63",
    # Azzouz-Nandi Spectral & Phase Features
    "gamma_max",
    "sigma_ap",
    "sigma_dp",
    "sigma_aa",
    "sigma_af",
    # Statistical moments of Envelope
    "env_mean",
    "env_var",
    "env_skew",
    "env_kurt",
    # Statistical moments of I and Q
    "i_var",
    "i_skew",
    "i_kurt",
    "q_var",
    "q_skew",
    "q_kurt",
    "iq_corr",
]


def compute_cumulants(x: np.ndarray) -> Dict[str, float]:
    """Computes second, fourth, and sixth-order cumulants of zero-mean complex signal.
    
    Args:
        x: 1D complex zero-mean numpy array
        
    Returns:
        Dictionary of raw and normalized cumulant values.
    """
    N = len(x)
    if N == 0:
        return {name: 0.0 for name in [
            "abs_C20", "abs_C21", "abs_C40", "abs_C41", "abs_C42", "abs_C63",
            "norm_C40", "norm_C42", "norm_C63"
        ]}

    x_conj = np.conj(x)

    # Second-order moments
    M20 = np.mean(x ** 2)
    M21 = np.mean(np.abs(x) ** 2)

    # Fourth-order moments
    M40 = np.mean(x ** 4)
    M41 = np.mean((x ** 3) * x_conj)
    M42 = np.mean(np.abs(x) ** 4)

    # Sixth-order moments
    M63 = np.mean(np.abs(x) ** 6)

    # Cumulant calculations (Swami & Mendel relations)
    C20 = M20
    C21 = M21

    C40 = M40 - 3.0 * (M20 ** 2)
    C41 = M41 - 3.0 * M20 * np.conj(M21)
    C42 = M42 - (np.abs(M20) ** 2) - 2.0 * (M21 ** 2)

    # Standardized C63 expression for zero-mean processes
    C63 = M63 - 9.0 * M42 * M21 + 12.0 * (M21 ** 3) - 3.0 * (np.abs(M20) ** 2) * M21

    c21_safe = np.real(C21) if np.real(C21) > 1e-12 else 1e-12

    # Scale-invariant normalized cumulants
    norm_C40 = float(np.abs(C40) / (c21_safe ** 2))
    norm_C42 = float(np.abs(C42) / (c21_safe ** 2))
    norm_C63 = float(np.abs(C63) / (c21_safe ** 3))

    return {
        "abs_C20": float(np.abs(C20)),
        "abs_C21": float(np.abs(C21)),
        "abs_C40": float(np.abs(C40)),
        "abs_C41": float(np.abs(C41)),
        "abs_C42": float(np.abs(C42)),
        "abs_C63": float(np.abs(C63)),
        "norm_C40": norm_C40,
        "norm_C42": norm_C42,
        "norm_C63": norm_C63,
    }


def compute_spectral_and_instantaneous_features(
    i_sig: np.ndarray,
    q_sig: np.ndarray
) -> Dict[str, float]:
    """Computes Azzouz-Nandi spectral, phase, and frequency parameters.
    
    Args:
        i_sig: 1D In-phase signal
        q_sig: 1D Quadrature signal
        
    Returns:
        Dictionary of instantaneous and spectral features.
    """
    N = len(i_sig)
    comp_sig = i_sig + 1j * q_sig

    # 1. Instantaneous amplitude / envelope
    a = np.abs(comp_sig)
    ma = np.mean(a)
    ma_safe = ma if ma > 1e-12 else 1e-12

    # Normalized centered instantaneous amplitude
    an = (a / ma_safe) - 1.0

    # gamma_max: Max normalized spectral power density of an
    dft_an = np.fft.fft(an)
    gamma_max = float(np.max(np.abs(dft_an) ** 2) / N)

    # sigma_aa: standard deviation of absolute centered amplitude
    sigma_aa = float(np.std(np.abs(an)))

    # 2. Instantaneous phase & non-linear phase component
    phase = np.unwrap(np.angle(comp_sig))

    # Detrend linear phase (frequency offset removal)
    t = np.arange(N)
    poly = np.polyfit(t, phase, 1)
    linear_trend = np.polyval(poly, t)
    phase_nl = phase - linear_trend

    # sigma_ap: std of absolute non-linear phase
    sigma_ap = float(np.std(np.abs(phase_nl)))

    # sigma_dp: std of direct non-linear phase
    sigma_dp = float(np.std(phase_nl))

    # 3. Instantaneous frequency
    inst_freq = np.diff(phase) / (2.0 * np.pi)
    if len(inst_freq) > 0:
        mf = np.mean(inst_freq)
        fn = inst_freq - mf
        sigma_af = float(np.std(np.abs(fn)))
    else:
        sigma_af = 0.0

    return {
        "gamma_max": gamma_max,
        "sigma_ap": sigma_ap,
        "sigma_dp": sigma_dp,
        "sigma_aa": sigma_aa,
        "sigma_af": sigma_af,
    }


def compute_statistical_moments(
    i_sig: np.ndarray,
    q_sig: np.ndarray
) -> Dict[str, float]:
    """Computes statistical distribution moments (variance, skewness, kurtosis, correlation).
    
    Args:
        i_sig: 1D In-phase signal
        q_sig: 1D Quadrature signal
        
    Returns:
        Dictionary of statistical features.
    """
    env = np.sqrt(i_sig ** 2 + q_sig ** 2)

    # Envelope moments
    env_mean = float(np.mean(env))
    env_var = float(np.var(env))
    env_skew = float(stats.skew(env)) if env_var > 1e-12 else 0.0
    env_kurt = float(stats.kurtosis(env)) if env_var > 1e-12 else 0.0

    # I moments
    i_var = float(np.var(i_sig))
    i_skew = float(stats.skew(i_sig)) if i_var > 1e-12 else 0.0
    i_kurt = float(stats.kurtosis(i_sig)) if i_var > 1e-12 else 0.0

    # Q moments
    q_var = float(np.var(q_sig))
    q_skew = float(stats.skew(q_sig)) if q_var > 1e-12 else 0.0
    q_kurt = float(stats.kurtosis(q_sig)) if q_var > 1e-12 else 0.0

    # I/Q Cross-Correlation
    denom = np.sqrt(i_var * q_var)
    if denom > 1e-12:
        cov = np.mean((i_sig - np.mean(i_sig)) * (q_sig - np.mean(q_sig)))
        iq_corr = float(np.clip(cov / denom, -1.0, 1.0))
    else:
        iq_corr = 0.0

    return {
        "env_mean": env_mean,
        "env_var": env_var,
        "env_skew": env_skew,
        "env_kurt": env_kurt,
        "i_var": i_var,
        "i_skew": i_skew,
        "i_kurt": i_kurt,
        "q_var": q_var,
        "q_skew": q_skew,
        "q_kurt": q_kurt,
        "iq_corr": iq_corr,
    }


def extract_features_single(iq_sample: np.ndarray) -> np.ndarray:
    """Extracts the full feature vector from a single (2, sequence_length) I/Q frame.
    
    Args:
        iq_sample: Array of shape (2, sequence_length)
        
    Returns:
        1D float32 array of shape (num_features,)
    """
    i_sig = iq_sample[0]
    q_sig = iq_sample[1]
    comp_sig = i_sig + 1j * q_sig
    zero_mean_sig = comp_sig - np.mean(comp_sig)

    # 1. Cumulants
    cum_dict = compute_cumulants(zero_mean_sig)

    # 2. Spectral & Phase
    spec_dict = compute_spectral_and_instantaneous_features(i_sig, q_sig)

    # 3. Statistical moments
    stat_dict = compute_statistical_moments(i_sig, q_sig)

    # Combine in deterministic order
    combined = {**cum_dict, **spec_dict, **stat_dict}
    vector = np.array([combined[k] for k in FEATURE_NAMES], dtype=np.float32)

    # Clean any NaNs or Infs
    vector = np.nan_to_num(vector, nan=0.0, posinf=1e6, neginf=-1e6)
    return vector


def extract_features_dataset(signals: np.ndarray) -> np.ndarray:
    """Extracts features for all signal frames in dataset.
    
    Args:
        signals: Array of shape (N_samples, 2, sequence_length)
        
    Returns:
        Array of shape (N_samples, num_features)
    """
    num_samples = signals.shape[0]
    num_feats = len(FEATURE_NAMES)
    features = np.zeros((num_samples, num_feats), dtype=np.float32)

    for i in range(num_samples):
        features[i] = extract_features_single(signals[i])

    return features
