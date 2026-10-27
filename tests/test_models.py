"""Unit tests for Baseline Machine Learning and Deep Learning Models."""

import unittest
import numpy as np
import torch
from src.baseline_models import BaselineModelSuite
from src.deep_models import CLDNN, AMCResNet1D, get_deep_model
from src.feature_extraction import FEATURE_NAMES


class TestModels(unittest.TestCase):

    def setUp(self):
        # Create synthetic feature dataset for baseline models
        rng = np.random.default_rng(42)
        self.n_samples = 60
        self.n_features = len(FEATURE_NAMES)
        self.n_classes = 4

        # Separate cluster centers for classes so models can learn
        self.X_train = rng.normal(0, 1, size=(self.n_samples, self.n_features)).astype(np.float32)
        self.y_train = rng.integers(0, self.n_classes, size=self.n_samples)
        for c in range(self.n_classes):
            mask = (self.y_train == c)
            self.X_train[mask] += c * 3.0

        self.X_test = self.X_train[:15]
        self.y_test = self.y_train[:15]
        self.snrs_test = rng.choice([0.0, 10.0], size=15)
        self.class_names = ["BPSK", "QPSK", "8PSK", "16QAM"]

    def test_baseline_suite_train_and_evaluate(self):
        """Verify that all baseline models (SVM, Random Forest, KNN) train and predict."""
        suite = BaselineModelSuite(random_state=42)
        suite.train_all(self.X_train, self.y_train)

        results = suite.evaluate_all(
            self.X_test,
            self.y_test,
            snrs_test=self.snrs_test,
            class_names=self.class_names
        )

        for name in ["SVM", "Random_Forest", "KNN"]:
            self.assertIn(name, results)
            self.assertIn("overall_accuracy", results[name])
            self.assertGreaterEqual(results[name]["overall_accuracy"], 0.0)
            self.assertLessEqual(results[name]["overall_accuracy"], 1.0)
            self.assertIn("snr_accuracies", results[name])

    def test_cldnn_forward_and_backward(self):
        """Verify CLDNN deep model forward pass shape, loss calculation, and backward pass."""
        model = CLDNN(num_classes=4, in_channels=2, conv_filters=16, lstm_hidden=16)
        x = torch.randn(8, 2, 128)  # Batch=8, Channels=2, Seq_Len=128
        target = torch.randint(0, 4, (8,))

        # Forward pass
        logits = model(x)
        self.assertEqual(logits.shape, (8, 4))

        # Backward pass
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, target)
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_resnet1d_forward(self):
        """Verify AMCResNet1D forward pass output shape."""
        model = AMCResNet1D(num_classes=4, in_channels=2, base_channels=16)
        x = torch.randn(4, 2, 128)
        logits = model(x)
        self.assertEqual(logits.shape, (4, 4))

    def test_model_factory(self):
        """Verify factory function instantiates correct model types."""
        m1 = get_deep_model("CLDNN", num_classes=4)
        self.assertIsInstance(m1, CLDNN)
        m2 = get_deep_model("ResNet", num_classes=4)
        self.assertIsInstance(m2, AMCResNet1D)


if __name__ == "__main__":
    unittest.main()
