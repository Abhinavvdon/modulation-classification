"""Unit tests for Feature Extraction Module."""

import unittest
import numpy as np
from src.feature_extraction import (
    compute_cumulants,
    compute_spectral_and_instantaneous_features,
    compute_statistical_moments,
    extract_features_single,
    extract_features_dataset,
    FEATURE_NAMES
)


class TestFeatureExtraction(unittest.TestCase):

    def test_cumulants_structure(self):
        """Verify cumulants output dictionary keys and valid float values."""
        rng = np.random.default_rng(42)
        # Generate random complex Gaussian noise
        noise = rng.normal(0, 1, 1000) + 1j * rng.normal(0, 1, 1000)
        cum_dict = compute_cumulants(noise)

        expected_keys = [
            "abs_C20", "abs_C21", "abs_C40", "abs_C41", "abs_C42", "abs_C63",
            "norm_C40", "norm_C42", "norm_C63"
        ]
        for k in expected_keys:
            self.assertIn(k, cum_dict)
            self.assertFalse(np.isnan(cum_dict[k]), f"NaN found in cumulant {k}")

    def test_cumulant_theoretical_discrimination(self):
        """Verify C42 / C21 discrimination between constant-modulus PSK and multi-modulus QAM."""
        rng = np.random.default_rng(42)
        # Perfect constant-modulus signal (PSK) -> |s| == 1
        phases = rng.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=5000)
        psk = np.exp(1j * phases)
        psk_cum = compute_cumulants(psk)

        # Multi-modulus 16-QAM constellation points
        grid = np.array([-3, -1, 1, 3])
        re = rng.choice(grid, size=5000)
        im = rng.choice(grid, size=5000)
        qam = (re + 1j * im) / np.sqrt(10.0)
        qam_cum = compute_cumulants(qam)

        # For PSK, norm_C42 = |M42 - |M20|^2 - 2 M21^2| / M21^2
        # Theoretical: PSK has constant amplitude, QAM has fluctuating amplitude.
        # norm_C42 for PSK is |-1| = 1.0; for 16-QAM it is ~0.68.
        self.assertNotEqual(round(psk_cum["norm_C42"], 2), round(qam_cum["norm_C42"], 2))

    def test_spectral_features(self):
        """Verify Azzouz-Nandi features calculation on pure sinusoidal vs noise."""
        t = np.linspace(0, 1, 256)
        i_sig = np.cos(2 * np.pi * 10 * t)
        q_sig = np.sin(2 * np.pi * 10 * t)
        spec_dict = compute_spectral_and_instantaneous_features(i_sig, q_sig)

        for k, v in spec_dict.items():
            self.assertFalse(np.isnan(v), f"NaN found in {k}")

    def test_feature_vector_extraction(self):
        """Verify complete feature vector length and finite bounds."""
        rng = np.random.default_rng(42)
        iq_sample = rng.normal(0, 1, size=(2, 128)).astype(np.float32)
        feats = extract_features_single(iq_sample)

        self.assertEqual(len(feats), len(FEATURE_NAMES))
        self.assertFalse(np.isnan(feats).any())
        self.assertFalse(np.isinf(feats).any())

    def test_dataset_feature_extraction(self):
        """Verify batch extraction dimensions."""
        signals = np.random.randn(10, 2, 128).astype(np.float32)
        features = extract_features_dataset(signals)
        self.assertEqual(features.shape, (10, len(FEATURE_NAMES)))


if __name__ == "__main__":
    unittest.main()
