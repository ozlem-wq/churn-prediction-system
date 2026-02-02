"""
Model Trainer
=============

Handles model training with multiple algorithms:
- Logistic Regression (baseline)
- Random Forest
- XGBoost (primary)

Includes hyperparameter tuning and cross-validation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.config import settings


class ModelTrainer:
    """Train and tune churn prediction models."""

    def __init__(self, model_dir: str = "models"):
        """
        Initialize trainer.

        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Any] = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance: Optional[pd.DataFrame] = None

    def get_models(self) -> Dict[str, Any]:
        """
        Get dictionary of model instances.

        Returns:
            Dictionary with model name and instance
        """
        return {
            "logistic_regression": LogisticRegression(
                random_state=settings.random_state,
                max_iter=1000,
                class_weight="balanced",
            ),
            "random_forest": RandomForestClassifier(
                random_state=settings.random_state,
                n_estimators=100,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "xgboost": XGBClassifier(
                random_state=settings.random_state,
                n_estimators=100,
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=1,  # Will be adjusted based on class imbalance
            ),
        }

    def get_param_grids(self) -> Dict[str, Dict]:
        """
        Get hyperparameter grids for each model.

        Returns:
            Dictionary with model name and parameter grid
        """
        return {
            "logistic_regression": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        }

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str = "xgboost",
        tune_hyperparameters: bool = False,
    ) -> Any:
        """
        Train a single model.

        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of model to train
            tune_hyperparameters: Whether to perform grid search

        Returns:
            Trained model instance
        """
        models = self.get_models()

        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

        model = models[model_name]

        # Adjust for class imbalance in XGBoost
        if model_name == "xgboost":
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            model.set_params(scale_pos_weight=scale_pos_weight)

        if tune_hyperparameters:
            param_grid = self.get_param_grids()[model_name]
            cv = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_state)

            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X_train, y_train)

            model = grid_search.best_estimator_
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            model.fit(X_train, y_train)

        self.models[model_name] = model
        return model

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool = False,
    ) -> Dict[str, Any]:
        """
        Train all available models.

        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform grid search

        Returns:
            Dictionary of trained models
        """
        for model_name in self.get_models().keys():
            print(f"\nTraining {model_name}...")
            self.train_model(X_train, y_train, model_name, tune_hyperparameters)

        return self.models

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "xgboost",
        scoring: str = "roc_auc",
    ) -> Dict[str, float]:
        """
        Perform cross-validation for a model.

        Args:
            X: Features
            y: Labels
            model_name: Name of model to validate
            scoring: Scoring metric

        Returns:
            Dictionary with CV results
        """
        model = self.models.get(model_name)
        if model is None:
            model = self.get_models()[model_name]

        cv = StratifiedKFold(
            n_splits=settings.cv_folds,
            shuffle=True,
            random_state=settings.random_state,
        )

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores.tolist(),
            "cv_folds": settings.cv_folds,
        }

    def compare_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Compare all trained models using cross-validation.

        Args:
            X: Features
            y: Labels

        Returns:
            DataFrame with model comparison results
        """
        results = []

        for model_name in self.models.keys():
            cv_results = self.cross_validate(X, y, model_name)
            results.append({
                "model": model_name,
                "mean_roc_auc": cv_results["mean_score"],
                "std_roc_auc": cv_results["std_score"],
            })

        comparison_df = pd.DataFrame(results).sort_values("mean_roc_auc", ascending=False)

        # Set best model
        self.best_model_name = comparison_df.iloc[0]["model"]
        self.best_model = self.models[self.best_model_name]

        return comparison_df

    def get_feature_importance(
        self,
        feature_names: List[str],
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            feature_names: List of feature names
            model_name: Model to get importance from (default: best model)

        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            model_name = self.best_model_name or "xgboost"

        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not trained")

        # Get importance based on model type
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        else:
            raise ValueError(f"Model {model_name} doesn't support feature importance")

        self.feature_importance = pd.DataFrame({
            "feature": feature_names[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return self.feature_importance

    def save_model(
        self,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Path:
        """
        Save trained model to disk.

        Args:
            model_name: Name of model to save (default: best model)
            version: Model version string

        Returns:
            Path to saved model file
        """
        if model_name is None:
            model_name = self.best_model_name or "xgboost"

        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not trained")

        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{model_name}_{version}.pkl"
        filepath = self.model_dir / filename

        joblib.dump(model, filepath)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "version": version,
            "saved_at": datetime.now().isoformat(),
            "model_type": type(model).__name__,
        }

        metadata_path = self.model_dir / f"{model_name}_{version}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return filepath

    def load_model(self, filepath: str) -> Any:
        """
        Load model from disk.

        Args:
            filepath: Path to model file

        Returns:
            Loaded model instance
        """
        model = joblib.load(filepath)
        return model
