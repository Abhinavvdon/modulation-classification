"""Unit tests for Signal Generation Module."""

import unittest
import numpy as np
from src.config import ModulationType, SignalConfig, ChannelConfig, ExperimentConfig
from src.signal_generator import (
    get_constellation,
    rrc_filter,
    add_awgn,
    generate_single_sample,
    generate_dataset
)


class TestSignalGenerator(unittest.TestCase):

    def test_constellations(self):
        """Verify constellation alphabet length and unit average energy."""
        test_cases = [
            (ModulationType.BPSK, 2),
            (ModulationType.QPSK, 4),
            (ModulationType.PSK8, 8),
            (ModulationType.QAM16, 16),
            (ModulationType.QAM64, 64),
        ]
        for mod, expected_len in test_cases:
            c = get_constellation(mod)
            self.assertEqual(len(c), expected_len, f"Wrong constellation length for {mod}")
            avg_energy = np.mean(np.abs(c) ** 2)
            self.assertAlmostEqual(avg_energy, 1.0, places=5, msg=f"Constellation {mod} not normalized to 1.0 energy")

    def test_rrc_filter_symmetry_and_energy(self):
        """Verify RRC filter taps are symmetric and have unit energy."""
        sps = 8
        span = 6
        alpha = 0.35
        h = rrc_filter(sps=sps, span=span, alpha=alpha)

        # Check length
        self.assertEqual(len(h), span * sps + 1)
        # Check symmetry
        np.testing.assert_allclose(h, h[::-1], atol=1e-10, err_msg="RRC filter is not symmetric")
        # Check unit energy sum(h^2) == 1.0
        self.assertAlmostEqual(np.sum(h ** 2), 1.0, places=5)

    def test_awgn_snr(self):
        """Verify that AWGN addition satisfies target SNR within statistical confidence."""
        rng = np.random.default_rng(1234)
        N = 100000
        # Unit energy pure sine tone
        clean = np.exp(1j * 2 * np.pi * 0.1 * np.arange(N))
        target_snr_db = 10.0

        noisy = add_awgn(clean, target_snr_db, rng)
        noise = noisy - clean
        sig_pow = np.mean(np.abs(clean) ** 2)
        noise_pow = np.mean(np.abs(noise) ** 2)
        measured_snr_db = 10 * np.log10(sig_pow / noise_pow)

        self.assertAlmostEqual(measured_snr_db, target_snr_db, delta=0.5)

    def test_single_sample_generation(self):
        """Verify single I/Q frame shape, finiteness, and absence of NaNs."""
        rng = np.random.default_rng(42)
        sig_cfg = SignalConfig(sequence_length=128, samples_per_symbol=8)
        chan_cfg = ChannelConfig(channel_type="awgn")

        sample = generate_single_sample(ModulationType.QPSK, 10.0, sig_cfg, chan_cfg, rng)
        self.assertEqual(sample.shape, (2, 128))
        self.assertFalse(np.isnan(sample).any(), "Sample contains NaN")
        self.assertFalse(np.isinf(sample).any(), "Sample contains Inf")

    def test_dataset_generation(self):
        """Verify mini dataset generator output shapes and label sets."""
        sig_cfg = SignalConfig(sequence_length=64, samples_per_snr_per_mod=5)
        chan_cfg = ChannelConfig(snr_range_db=[0.0, 10.0], channel_type="awgn")
        exp_cfg = ExperimentConfig(
            signal=sig_cfg,
            channel=chan_cfg,
            modulations=[ModulationType.BPSK, ModulationType.QPSK]
        )

        signals, labels, snrs, class_names = generate_dataset(exp_cfg)

        expected_count = 2 * 2 * 5  # 2 mods * 2 snrs * 5 samples = 20
        self.assertEqual(signals.shape, (expected_count, 2, 64))
        self.assertEqual(labels.shape, (expected_count,))
        self.assertEqual(snrs.shape, (expected_count,))
        self.assertEqual(class_names, ["BPSK", "QPSK"])


if __name__ == "__main__":
    unittest.main()
