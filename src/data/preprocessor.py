"""
Data Preprocessor
=================

Handles data cleaning and transformation:
- Missing value imputation
- Categorical encoding
- Feature scaling
- Data validation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """Preprocess data for machine learning."""

    def __init__(self):
        """Initialize preprocessor with encoders and scalers."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.fitted = False

        # Column definitions
        self.categorical_columns = [
            "gender",
            "partner",
            "dependents",
            "phone_service",
            "multiple_lines",
            "internet_service",
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
            "contract_type",
            "paperless_billing",
            "payment_method",
        ]

        self.numeric_columns = [
            "tenure_months",
            "monthly_charges",
            "total_charges",
        ]

        self.boolean_columns = [
            "senior_citizen",
        ]

        self.target_column = "churned"

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Handle TotalCharges - convert to numeric, handle empty strings
        if "total_charges" in df.columns:
            df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

            # Fill missing total_charges with monthly_charges * tenure
            if "monthly_charges" in df.columns and "tenure_months" in df.columns:
                mask = df["total_charges"].isna()
                df.loc[mask, "total_charges"] = (
                    df.loc[mask, "monthly_charges"] * df.loc[mask, "tenure_months"]
                )

            # Fill any remaining NaN with 0
            df["total_charges"] = df["total_charges"].fillna(0)

        # Convert Yes/No to boolean where applicable
        yes_no_columns = ["partner", "dependents", "phone_service", "paperless_billing"]
        for col in yes_no_columns:
            if col in df.columns and df[col].dtype == "object":
                df[col] = df[col].map({"Yes": True, "No": False})

        # Convert Churn to boolean
        if self.target_column in df.columns:
            if df[self.target_column].dtype == "object":
                df[self.target_column] = df[self.target_column].map({"Yes": True, "No": False})

        # Ensure senior_citizen is boolean
        if "senior_citizen" in df.columns:
            df["senior_citizen"] = df["senior_citizen"].astype(bool)

        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "median",
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.

        Args:
            df: DataFrame with potential missing values
            strategy: Imputation strategy ('mean', 'median', 'mode', 'zero')

        Returns:
            DataFrame with imputed values
        """
        df = df.copy()

        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue

            if df[col].dtype in ["int64", "float64"]:
                if strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "zero":
                    df[col] = df[col].fillna(0)
            else:
                # Categorical columns - fill with mode
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown")

        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: DataFrame with categorical columns
            fit: Whether to fit encoders (True for training)

        Returns:
            DataFrame with encoded categories
        """
        df = df.copy()

        for col in self.categorical_columns:
            if col not in df.columns:
                continue

            if fit:
                le = LabelEncoder()
                # Fit on all possible values
                df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df[f"{col}_encoded"] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    raise ValueError(f"Encoder not fitted for column: {col}")

        return df

    def scale_numeric(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Scale numeric features.

        Args:
            df: DataFrame with numeric columns
            fit: Whether to fit scaler (True for training)

        Returns:
            DataFrame with scaled numeric columns
        """
        df = df.copy()

        # Get numeric columns that exist in DataFrame
        cols_to_scale = [c for c in self.numeric_columns if c in df.columns]

        if not cols_to_scale:
            return df

        if fit:
            self.scaler = StandardScaler()
            scaled_values = self.scaler.fit_transform(df[cols_to_scale])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            scaled_values = self.scaler.transform(df[cols_to_scale])

        # Add scaled columns
        for i, col in enumerate(cols_to_scale):
            df[f"{col}_scaled"] = scaled_values[:, i]

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessors and transform data.

        Args:
            df: Raw DataFrame

        Returns:
            Preprocessed DataFrame
        """
        df = self.clean_data(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df, fit=True)
        df = self.scale_numeric(df, fit=True)
        self.fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessors.

        Args:
            df: Raw DataFrame

        Returns:
            Preprocessed DataFrame
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        df = self.clean_data(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df, fit=False)
        df = self.scale_numeric(df, fit=False)
        return df

    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature column names for model training.

        Returns:
            List of feature column names
        """
        features = []

        # Encoded categorical columns
        for col in self.categorical_columns:
            features.append(f"{col}_encoded")

        # Scaled numeric columns
        for col in self.numeric_columns:
            features.append(f"{col}_scaled")

        # Boolean columns
        features.extend(self.boolean_columns)

        return features

    def prepare_for_training(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Tuple of (feature DataFrame, target Series)
        """
        feature_cols = self.get_feature_columns()

        # Filter to columns that exist
        available_cols = [c for c in feature_cols if c in df.columns]

        X = df[available_cols].copy()
        y = df[self.target_column].astype(int) if self.target_column in df.columns else None

        return X, y
