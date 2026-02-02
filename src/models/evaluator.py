"""
Model Evaluator
===============

Evaluates model performance using various metrics:
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC
- Confusion Matrix
- Classification Report
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelEvaluator:
    """Evaluate churn prediction models."""

    def __init__(self, threshold: float = 0.5):
        """
        Initialize evaluator.

        Args:
            threshold: Classification threshold for probabilities
        """
        self.threshold = threshold
        self.metrics: Dict[str, float] = {}
        self.confusion_matrix: Optional[np.ndarray] = None
        self.roc_data: Optional[Dict] = None
        self.pr_data: Optional[Dict] = None

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model with all metrics.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Optional custom threshold

        Returns:
            Dictionary with all metrics
        """
        if threshold is None:
            threshold = self.threshold

        # Convert probabilities to predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        self.metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "threshold": threshold,
        }

        # Confusion matrix
        self.confusion_matrix = confusion_matrix(y_true, y_pred)

        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        self.roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
            "auc": self.metrics["roc_auc"],
        }

        # Precision-Recall curve data
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        self.pr_data = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
            "auc": auc(recall, precision),
        }

        self.metrics["pr_auc"] = self.pr_data["auc"]

        return self.metrics

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: Optional[float] = None,
    ) -> str:
        """
        Get detailed classification report.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Optional custom threshold

        Returns:
            Classification report string
        """
        if threshold is None:
            threshold = self.threshold

        y_pred = (y_pred_proba >= threshold).astype(int)

        return classification_report(
            y_true,
            y_pred,
            target_names=["No Churn", "Churn"],
        )

    def get_confusion_matrix_dict(self) -> Dict[str, int]:
        """
        Get confusion matrix as dictionary.

        Returns:
            Dictionary with TN, FP, FN, TP counts
        """
        if self.confusion_matrix is None:
            raise ValueError("Run evaluate() first")

        tn, fp, fn, tp = self.confusion_matrix.ravel()

        return {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = "f1",
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'recall', 'precision')

        Returns:
            Tuple of (optimal_threshold, best_score)
        """
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_score = 0
        best_threshold = 0.5

        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)

            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = thresh

        return best_threshold, best_score

    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            y_true: True labels
            predictions: Dict with model name -> predicted probabilities

        Returns:
            DataFrame with comparison results
        """
        results = []

        for model_name, y_pred_proba in predictions.items():
            metrics = self.evaluate(y_true, y_pred_proba)
            metrics["model"] = model_name
            results.append(metrics)

        return pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    def get_performance_at_thresholds(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Get performance metrics at different thresholds.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            thresholds: List of thresholds to evaluate

        Returns:
            DataFrame with metrics at each threshold
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        results = []
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)

            results.append({
                "threshold": thresh,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "predicted_churns": y_pred.sum(),
                "actual_churns": y_true.sum(),
            })

        return pd.DataFrame(results)

    def get_summary(self) -> Dict:
        """
        Get evaluation summary.

        Returns:
            Dictionary with all evaluation results
        """
        if not self.metrics:
            raise ValueError("Run evaluate() first")

        return {
            "metrics": self.metrics,
            "confusion_matrix": self.get_confusion_matrix_dict(),
            "roc_auc": self.roc_data["auc"] if self.roc_data else None,
            "pr_auc": self.pr_data["auc"] if self.pr_data else None,
        }

    def print_report(self) -> None:
        """Print formatted evaluation report."""
        if not self.metrics:
            print("No evaluation results. Run evaluate() first.")
            return

        print("\n" + "=" * 50)
        print("MODEL EVALUATION REPORT")
        print("=" * 50)

        print("\n📊 Performance Metrics:")
        print(f"  Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"  Precision: {self.metrics['precision']:.4f}")
        print(f"  Recall:    {self.metrics['recall']:.4f}")
        print(f"  F1-Score:  {self.metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {self.metrics['roc_auc']:.4f}")

        if "pr_auc" in self.metrics:
            print(f"  PR-AUC:    {self.metrics['pr_auc']:.4f}")

        print(f"\n🎯 Threshold: {self.metrics['threshold']}")

        if self.confusion_matrix is not None:
            cm = self.get_confusion_matrix_dict()
            print("\n📋 Confusion Matrix:")
            print(f"  True Negatives:  {cm['true_negatives']}")
            print(f"  False Positives: {cm['false_positives']}")
            print(f"  False Negatives: {cm['false_negatives']}")
            print(f"  True Positives:  {cm['true_positives']}")

        print("\n" + "=" * 50)
