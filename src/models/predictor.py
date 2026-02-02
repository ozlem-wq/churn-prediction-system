"""
Churn Predictor
===============

Makes predictions using trained models.
Provides probability scores and risk levels.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.data.preprocessor import DataPreprocessor
from src.features.engineer import FeatureEngineer


class ChurnPredictor:
    """Make churn predictions using trained models."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor: Optional[DataPreprocessor] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model file
            preprocessor: Fitted preprocessor instance
            feature_engineer: Feature engineer instance
        """
        self.model = None
        self.model_version = None
        self.model_name = None

        self.preprocessor = preprocessor or DataPreprocessor()
        self.feature_engineer = feature_engineer or FeatureEngineer()

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Load model from disk.

        Args:
            model_path: Path to model file (.pkl)
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(path)
        self.model_name = path.stem.split("_")[0]
        self.model_version = "_".join(path.stem.split("_")[1:])

    def predict(
        self,
        data: Union[pd.DataFrame, Dict],
        return_probability: bool = True,
    ) -> Dict:
        """
        Make prediction for a single customer.

        Args:
            data: Customer data as DataFrame or dictionary
            return_probability: Whether to return probability scores

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model first.")

        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Preprocess data
        df = self.feature_engineer.create_all_features(df)

        # Get probability
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(df)[:, 1]
        else:
            probabilities = self.model.predict(df)

        probability = float(probabilities[0])
        prediction = probability >= settings.churn_probability_threshold
        risk_level = settings.get_risk_level(probability)

        result = {
            "customer_id": df.get("customer_id", pd.Series(["unknown"])).iloc[0],
            "churn_probability": round(probability, 4),
            "prediction": prediction,
            "risk_level": risk_level,
            "model_version": self.model_version,
            "predicted_at": datetime.now().isoformat(),
        }

        if return_probability:
            result["top_risk_factors"] = self._get_risk_factors(df.iloc[0])

        return result

    def predict_batch(
        self,
        data: pd.DataFrame,
        return_dataframe: bool = True,
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        Make predictions for multiple customers.

        Args:
            data: DataFrame with customer data
            return_dataframe: Whether to return DataFrame or list

        Returns:
            DataFrame or list with predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model first.")

        df = data.copy()

        # Preprocess data
        df = self.feature_engineer.create_all_features(df)

        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(df)[:, 1]
        else:
            probabilities = self.model.predict(df)

        # Build results
        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                "customer_id": df.get("customer_id", pd.Series(range(len(df)))).iloc[i],
                "churn_probability": round(float(prob), 4),
                "prediction": prob >= settings.churn_probability_threshold,
                "risk_level": settings.get_risk_level(prob),
            })

        if return_dataframe:
            return pd.DataFrame(results)

        return results

    def _get_risk_factors(self, row: pd.Series) -> List[str]:
        """
        Identify top risk factors for a customer.

        Args:
            row: Series with customer data

        Returns:
            List of risk factor descriptions
        """
        risk_factors = []

        # Contract type risk
        contract = row.get("contract_type", "")
        if contract == "Month-to-month":
            risk_factors.append("Month-to-month contract (high churn risk)")

        # Tenure risk
        tenure = row.get("tenure_months", 0)
        if tenure <= 12:
            risk_factors.append(f"Short tenure ({tenure} months)")

        # No support services
        tech_support = row.get("tech_support", "")
        online_security = row.get("online_security", "")
        if tech_support != "Yes" and online_security != "Yes":
            risk_factors.append("No premium support services")

        # Payment method risk
        payment = row.get("payment_method", "")
        if payment == "Electronic check":
            risk_factors.append("Electronic check payment (higher churn rate)")

        # High monthly charges
        charges = row.get("monthly_charges", 0)
        if charges > 70:
            risk_factors.append(f"High monthly charges (${charges})")

        # Internet service type
        internet = row.get("internet_service", "")
        if internet == "Fiber optic":
            risk_factors.append("Fiber optic internet (higher churn correlation)")

        # Senior citizen
        senior = row.get("senior_citizen", False)
        if senior:
            risk_factors.append("Senior citizen customer segment")

        # No dependents or partner
        partner = row.get("partner", True)
        dependents = row.get("dependents", True)
        if not partner and not dependents:
            risk_factors.append("No partner or dependents")

        return risk_factors[:5]  # Return top 5 factors

    def get_model_info(self) -> Dict:
        """
        Get information about loaded model.

        Returns:
            Dictionary with model metadata
        """
        if self.model is None:
            return {"status": "No model loaded"}

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": type(self.model).__name__,
            "threshold": settings.churn_probability_threshold,
            "risk_levels": {
                "high": f">= {settings.high_risk_threshold}",
                "medium": f">= {settings.medium_risk_threshold}",
                "low": f"< {settings.medium_risk_threshold}",
            },
        }


class PredictorService:
    """Singleton service for production prediction."""

    _instance: Optional[ChurnPredictor] = None

    @classmethod
    def get_predictor(cls) -> ChurnPredictor:
        """Get or create predictor instance."""
        if cls._instance is None:
            cls._instance = ChurnPredictor()
            # Try to load default model
            model_path = Path(settings.model_path) / f"{settings.model_name}_latest.pkl"
            if model_path.exists():
                cls._instance.load_model(str(model_path))
        return cls._instance

    @classmethod
    def reload_model(cls, model_path: str) -> None:
        """Reload model from new path."""
        predictor = cls.get_predictor()
        predictor.load_model(model_path)
