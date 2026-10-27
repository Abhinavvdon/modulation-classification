"""Signal Generation Module for Automatic Modulation Classification (AMC).

Provides synthetic generation of baseband I/Q signals for digital modulations:
- BPSK (Binary Phase Shift Keying)
- QPSK (Quadrature Phase Shift Keying)
- 8PSK (8-ary Phase Shift Keying)
- 16QAM (16-ary Quadrature Amplitude Modulation)
- 64QAM (64-ary Quadrature Amplitude Modulation)

Includes pulse shaping (Root-Raised Cosine) and realistic wireless channel models:
- AWGN (Additive White Gaussian Noise) across variable SNR
- Multipath Fading (Rayleigh and Rician tapped-delay line models)
- Carrier Frequency Offset (CFO)
- Phase Noise and Timing Offsets
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from .config import ModulationType, SignalConfig, ChannelConfig, ExperimentConfig


def get_constellation(mod_type: ModulationType) -> np.ndarray:
    """Returns normalized unit-average-energy constellation alphabet for given modulation.
    
    Args:
        mod_type: ModulationType enum
        
    Returns:
        1D complex numpy array of constellation points with E[|s|^2] == 1.0.
    """
    if mod_type == ModulationType.BPSK:
        # BPSK: {-1, +1}
        c = np.array([-1.0, 1.0], dtype=np.complex128)
        return c / np.sqrt(np.mean(np.abs(c) ** 2))

    elif mod_type == ModulationType.QPSK:
        # QPSK: Gray-coded 4 points
        points = [
            1.0 + 1.0j,
            -1.0 + 1.0j,
            -1.0 - 1.0j,
            1.0 - 1.0j,
        ]
        c = np.array(points, dtype=np.complex128)
        return c / np.sqrt(np.mean(np.abs(c) ** 2))

    elif mod_type == ModulationType.PSK8:
        # 8-PSK: 8 uniformly spaced phases on unit circle
        angles = np.arange(8) * (2 * np.pi / 8)
        c = np.exp(1j * angles).astype(np.complex128)
        return c / np.sqrt(np.mean(np.abs(c) ** 2))

    elif mod_type == ModulationType.QAM16:
        # 16-QAM: 4x4 square grid {-3, -1, 1, 3} x {-3, -1, 1, 3}
        re_vals = np.array([-3.0, -1.0, 1.0, 3.0])
        im_vals = np.array([-3.0, -1.0, 1.0, 3.0])
        grid_re, grid_im = np.meshgrid(re_vals, im_vals)
        c = (grid_re.flatten() + 1j * grid_im.flatten()).astype(np.complex128)
        return c / np.sqrt(np.mean(np.abs(c) ** 2))

    elif mod_type == ModulationType.QAM64:
        # 64-QAM: 8x8 square grid
        re_vals = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
        im_vals = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
        grid_re, grid_im = np.meshgrid(re_vals, im_vals)
        c = (grid_re.flatten() + 1j * grid_im.flatten()).astype(np.complex128)
        return c / np.sqrt(np.mean(np.abs(c) ** 2))

    else:
        raise ValueError(f"Unsupported modulation type: {mod_type}")


def rrc_filter(
    sps: int,
    span: int,
    alpha: float
) -> np.ndarray:
    """Generates a square-root raised cosine (SRRC / RRC) pulse shaping filter.
    
    Args:
        sps: Samples per symbol (oversampling factor)
        span: Filter span in symbols (length of filter is span * sps + 1)
        alpha: Roll-off factor (0 < alpha <= 1)
        
    Returns:
        1D float64 numpy array of filter taps normalized to unit energy.
    """
    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1) / sps
    h = np.zeros(len(t), dtype=np.float64)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha > 0 and np.isclose(np.abs(ti), 1.0 / (4 * alpha)):
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            denom = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = num / denom

    # Normalize filter to unit energy sum(h^2) = 1
    h = h / np.sqrt(np.sum(h ** 2))
    return h


def apply_rrc_shaping(
    symbols: np.ndarray,
    sps: int,
    span: int = 8,
    alpha: float = 0.35
) -> np.ndarray:
    """Upsamples symbols and applies Root-Raised-Cosine (RRC) pulse shaping.
    
    Args:
        symbols: 1D complex array of symbols
        sps: Samples per symbol
        span: Filter span in symbols
        alpha: Roll-off factor
        
    Returns:
        Pulse-shaped complex continuous baseband signal.
    """
    # 1. Upsample by zero-insertion
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols

    # 2. Design RRC filter
    h = rrc_filter(sps=sps, span=span, alpha=alpha)

    # 3. Convolution
    filtered = np.convolve(upsampled, h, mode="same")
    return filtered


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Adds complex Additive White Gaussian Noise (AWGN) at specified SNR.
    
    Args:
        signal: 1D complex array
        snr_db: Target Signal-to-Noise Ratio in dB
        rng: NumPy random generator
        
    Returns:
        Noisy complex signal
    """
    sig_power = np.mean(np.abs(signal) ** 2)
    if sig_power <= 0:
        sig_power = 1.0

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear

    # Complex Gaussian noise: Real and Imag each have variance noise_power / 2
    noise_real = rng.normal(0, np.sqrt(noise_power / 2.0), len(signal))
    noise_imag = rng.normal(0, np.sqrt(noise_power / 2.0), len(signal))
    noise = noise_real + 1j * noise_imag

    return signal + noise


def apply_multipath_fading(
    signal: np.ndarray,
    delays: Tuple[int, ...],
    powers_db: Tuple[float, ...],
    channel_type: str,
    rician_k: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Applies tapped-delay line multipath channel fading (Rayleigh or Rician).
    
    Args:
        signal: 1D complex signal
        delays: Sample delays for each tap
        powers_db: Relative average power for each tap in dB
        channel_type: 'rayleigh', 'rician', or 'awgn'
        rician_k: K-factor for Rician fading (linear scale)
        rng: NumPy random generator
        
    Returns:
        Faded complex signal
    """
    if channel_type == "awgn" or len(delays) == 0:
        return signal

    # Convert tap powers from dB to linear
    linear_powers = 10.0 ** (np.array(powers_db) / 10.0)
    linear_powers /= np.sum(linear_powers)  # normalize total channel power to 1.0

    output = np.zeros(len(signal), dtype=np.complex128)

    for delay, power in zip(delays, linear_powers):
        if channel_type == "rician" and delay == 0:
            # First tap has dominant specular LOS component
            los_power = power * (rician_k / (rician_k + 1.0))
            scatter_power = power * (1.0 / (rician_k + 1.0))
            h_los = np.sqrt(los_power) * np.exp(1j * rng.uniform(0, 2 * np.pi))
            h_scat = (rng.normal(0, np.sqrt(scatter_power / 2.0)) + 
                      1j * rng.normal(0, np.sqrt(scatter_power / 2.0)))
            tap_coeff = h_los + h_scat
        else:
            # Rayleigh fading (diffuse scattered components)
            tap_coeff = (rng.normal(0, np.sqrt(power / 2.0)) + 
                         1j * rng.normal(0, np.sqrt(power / 2.0)))

        # Delayed signal
        if delay == 0:
            output += tap_coeff * signal
        elif delay < len(signal):
            delayed_sig = np.zeros_like(signal)
            delayed_sig[delay:] = signal[:-delay]
            output += tap_coeff * delayed_sig

    return output


def apply_cfo_and_phase_noise(
    signal: np.ndarray,
    sample_rate: float,
    cfo_max: float,
    phase_noise_std: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Simulates Carrier Frequency Offset (CFO) and oscillator phase noise / jitter.
    
    Args:
        signal: 1D complex signal
        sample_rate: Sampling frequency in Hz
        cfo_max: Max normalized frequency offset
        phase_noise_std: Phase noise standard deviation per step in radians
        rng: NumPy random generator
        
    Returns:
        Impaired complex signal
    """
    N = len(signal)
    # Carrier frequency offset: delta_f in [-cfo_max, cfo_max]
    cfo = rng.uniform(-cfo_max, cfo_max)
    t = np.arange(N)
    cfo_phasor = np.exp(1j * 2 * np.pi * cfo * t)

    # Initial random phase offset + Wiener phase noise walk
    initial_phase = rng.uniform(0, 2 * np.pi)
    phase_innovations = rng.normal(0, phase_noise_std, N)
    random_phase = initial_phase + np.cumsum(phase_innovations)
    phase_phasor = np.exp(1j * random_phase)

    return signal * cfo_phasor * phase_phasor


def generate_single_sample(
    mod_type: ModulationType,
    snr_db: float,
    signal_config: SignalConfig,
    channel_config: ChannelConfig,
    rng: np.random.Generator
) -> np.ndarray:
    """Generates a single I/Q frame of fixed sequence length under specified noise & channel.
    
    Args:
        mod_type: Target modulation scheme
        snr_db: Signal-to-Noise Ratio in dB
        signal_config: Signal parameters
        channel_config: Wireless channel parameters
        rng: NumPy random generator
        
    Returns:
        2D numpy array of shape (2, sequence_length) where row 0 is I and row 1 is Q.
    """
    constellation = get_constellation(mod_type)
    num_symbols_needed = int(np.ceil(signal_config.sequence_length / signal_config.samples_per_symbol)) + signal_config.filter_span_symbols + 4

    # 1. Random symbol sequence
    sym_indices = rng.integers(0, len(constellation), size=num_symbols_needed)
    symbols = constellation[sym_indices]

    # 2. RRC Pulse Shaping
    shaped_signal = apply_rrc_shaping(
        symbols=symbols,
        sps=signal_config.samples_per_symbol,
        span=signal_config.filter_span_symbols,
        alpha=signal_config.roll_off
    )

    # 3. Discard filter transient at the beginning and slice to sequence length
    transient = (signal_config.filter_span_symbols * signal_config.samples_per_symbol) // 2
    if len(shaped_signal) >= transient + signal_config.sequence_length:
        sliced_signal = shaped_signal[transient : transient + signal_config.sequence_length]
    else:
        sliced_signal = shaped_signal[: signal_config.sequence_length]

    # 4. Multipath Fading
    faded_signal = apply_multipath_fading(
        signal=sliced_signal,
        delays=channel_config.multipath_delays,
        powers_db=channel_config.multipath_powers_db,
        channel_type=channel_config.channel_type,
        rician_k=channel_config.rician_k,
        rng=rng
    )

    # 5. CFO & Phase Noise (if enabled)
    if channel_config.cfo_max > 0 or channel_config.phase_noise_std > 0:
        impaired_signal = apply_cfo_and_phase_noise(
            signal=faded_signal,
            sample_rate=signal_config.sample_rate,
            cfo_max=channel_config.cfo_max,
            phase_noise_std=channel_config.phase_noise_std,
            rng=rng
        )
    else:
        impaired_signal = faded_signal

    # 6. Add AWGN according to target SNR
    noisy_signal = add_awgn(impaired_signal, snr_db, rng)

    # 7. Normalize received signal power to unit energy E[|r|^2] = 1 for stability
    power = np.mean(np.abs(noisy_signal) ** 2)
    if power > 0:
        noisy_signal = noisy_signal / np.sqrt(power)

    # 8. Separate In-phase (I) and Quadrature (Q)
    i_component = np.real(noisy_signal).astype(np.float32)
    q_component = np.imag(noisy_signal).astype(np.float32)

    # Shape: (2, sequence_length)
    iq_frame = np.stack([i_component, q_component], axis=0)
    return iq_frame


def generate_dataset(
    experiment_config: ExperimentConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Generates the full synthetic AMC dataset across all modulations and SNR levels.
    
    Args:
        experiment_config: Complete experiment configuration
        
    Returns:
        signals: Array of shape (Total_Samples, 2, sequence_length)
        labels: 1D integer array of class indices (Total_Samples,)
        snrs: 1D float array of SNR values (Total_Samples,)
        class_names: List of string names for classes
    """
    rng = np.random.default_rng(experiment_config.signal.random_seed)
    modulations = experiment_config.modulations
    snr_range = experiment_config.channel.snr_range_db
    samples_per_snr = experiment_config.signal.samples_per_snr_per_mod

    total_samples = len(modulations) * len(snr_range) * samples_per_snr
    seq_len = experiment_config.signal.sequence_length

    signals = np.zeros((total_samples, 2, seq_len), dtype=np.float32)
    labels = np.zeros(total_samples, dtype=np.int64)
    snrs = np.zeros(total_samples, dtype=np.float32)

    idx = 0
    for mod_idx, mod in enumerate(modulations):
        for snr in snr_range:
            for _ in range(samples_per_snr):
                iq_sample = generate_single_sample(
                    mod_type=mod,
                    snr_db=snr,
                    signal_config=experiment_config.signal,
                    channel_config=experiment_config.channel,
                    rng=rng
                )
                signals[idx] = iq_sample
                labels[idx] = mod_idx
                snrs[idx] = snr
                idx += 1

    class_names = [m.value for m in modulations]
    return signals, labels, snrs, class_names
