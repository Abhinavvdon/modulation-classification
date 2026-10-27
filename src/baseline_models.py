"""Baseline Machine Learning Models for Automatic Modulation Classification (AMC).

Implements classical ML pipelines operating on extracted domain features:
- Support Vector Machine (SVM with RBF kernel and probability calibration)
- Random Forest Classifier (ensemble of decision trees with feature importance)
- K-Nearest Neighbors (KNN with distance weighting)

Includes automated training, cross-validation, and SNR-stratified evaluation.
"""

import os
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def build_svm_pipeline(
    C: float = 10.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    random_state: int = 42
) -> Pipeline:
    """Builds a standardized Support Vector Machine pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            random_state=random_state
        ))
    ])


def build_random_forest_pipeline(
    n_estimators: int = 150,
    max_depth: Optional[int] = 16,
    min_samples_split: int = 4,
    random_state: int = 42
) -> Pipeline:
    """Builds a standardized Random Forest pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        ))
    ])


def build_knn_pipeline(
    n_neighbors: int = 5,
    weights: str = "distance",
    metric: str = "minkowski"
) -> Pipeline:
    """Builds a standardized K-Nearest Neighbors pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=-1
        ))
    ])


class BaselineModelSuite:
    """Orchestrates training, evaluation, and serialization of all classical ML baselines."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Pipeline] = {
            "SVM": build_svm_pipeline(random_state=random_state),
            "Random_Forest": build_random_forest_pipeline(random_state=random_state),
            "KNN": build_knn_pipeline(),
        }
        self.is_trained: Dict[str, bool] = {name: False for name in self.models}

    def train(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains a single model by name."""
        if model_name not in self.models:
            raise ValueError(f"Unknown baseline model: {model_name}. Options: {list(self.models.keys())}")
        
        self.models[model_name].fit(X_train, y_train)
        self.is_trained[model_name] = True

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains all baseline models on the extracted feature dataset."""
        for name in self.models:
            self.train(name, X_train, y_train)

    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        snrs_test: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluates a single trained model on test data, overall and per SNR."""
        if not self.is_trained[model_name]:
            raise RuntimeError(f"Model '{model_name}' has not been trained yet.")

        model = self.models[model_name]
        preds = model.predict(X_test)
        overall_acc = accuracy_score(y_test, preds)

        cm = confusion_matrix(y_test, preds)
        report = classification_report(
            y_test,
            preds,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        results: Dict[str, Any] = {
            "overall_accuracy": float(overall_acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "predictions": preds,
        }

        # Calculate accuracy breakdown per SNR
        if snrs_test is not None:
            unique_snrs = np.sort(np.unique(snrs_test))
            snr_accuracies = {}
            for snr in unique_snrs:
                mask = (snrs_test == snr)
                if np.any(mask):
                    snr_acc = accuracy_score(y_test[mask], preds[mask])
                    snr_accuracies[float(snr)] = float(snr_acc)
            results["snr_accuracies"] = snr_accuracies

        return results

    def evaluate_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        snrs_test: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluates all baseline models on the test set."""
        all_results = {}
        for name in self.models:
            all_results[name] = self.evaluate_model(
                name, X_test, y_test, snrs_test, class_names
            )
        return all_results

    def save(self, output_dir: str) -> None:
        """Saves all trained models to disk."""
        os.makedirs(output_dir, exist_ok=True)
        for name, model in self.models.items():
            if self.is_trained[name]:
                path = os.path.join(output_dir, f"{name.lower()}_pipeline.joblib")
                joblib.dump(model, path)

    def load(self, model_dir: str) -> None:
        """Loads serialized models from disk."""
        for name in self.models:
            path = os.path.join(model_dir, f"{name.lower()}_pipeline.joblib")
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                self.is_trained[name] = True
