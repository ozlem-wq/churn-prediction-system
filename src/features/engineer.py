"""
Feature Engineer
================

Creates derived features from raw customer data:
- Tenure grouping
- Service aggregations
- Risk scores
- Interaction features
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import settings


class FeatureEngineer:
    """Create derived features for churn prediction."""

    def __init__(self):
        """Initialize feature engineer."""
        self.tenure_bins = settings.tenure_bins
        self.tenure_labels = settings.tenure_labels

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all derived features.

        Args:
            df: DataFrame with raw customer data

        Returns:
            DataFrame with additional feature columns
        """
        df = df.copy()

        df = self.create_tenure_features(df)
        df = self.create_charge_features(df)
        df = self.create_service_features(df)
        df = self.create_risk_features(df)
        df = self.create_interaction_features(df)

        return df

    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-related features.

        Features:
            - tenure_group: Binned tenure category
            - is_new_customer: Customer with tenure <= 6 months
            - is_loyal_customer: Customer with tenure >= 48 months

        Args:
            df: DataFrame with tenure_months column

        Returns:
            DataFrame with tenure features
        """
        df = df.copy()

        if "tenure_months" not in df.columns:
            return df

        # Tenure grouping
        df["tenure_group"] = pd.cut(
            df["tenure_months"],
            bins=self.tenure_bins + [np.inf],
            labels=self.tenure_labels,
            right=True,
        )

        # New customer flag (first 6 months)
        df["is_new_customer"] = (df["tenure_months"] <= 6).astype(int)

        # Loyal customer flag (4+ years)
        df["is_loyal_customer"] = (df["tenure_months"] >= 48).astype(int)

        # Tenure in years
        df["tenure_years"] = df["tenure_months"] / 12

        return df

    def create_charge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create charge-related features.

        Features:
            - avg_monthly_charge: Total charges / tenure
            - charge_per_service: Monthly charges / service count
            - is_high_value: Monthly charges > 70

        Args:
            df: DataFrame with charge columns

        Returns:
            DataFrame with charge features
        """
        df = df.copy()

        # Average monthly charge
        if "total_charges" in df.columns and "tenure_months" in df.columns:
            df["avg_monthly_charge"] = np.where(
                df["tenure_months"] > 0,
                df["total_charges"] / df["tenure_months"],
                df.get("monthly_charges", 0),
            )
        elif "monthly_charges" in df.columns:
            df["avg_monthly_charge"] = df["monthly_charges"]

        # High value customer flag
        if "monthly_charges" in df.columns:
            df["is_high_value"] = (df["monthly_charges"] > 70).astype(int)

            # Charge tier
            df["charge_tier"] = pd.cut(
                df["monthly_charges"],
                bins=[0, 30, 60, 90, np.inf],
                labels=["Low", "Medium", "High", "Premium"],
            )

        # Lifetime value estimate
        if "monthly_charges" in df.columns and "tenure_months" in df.columns:
            df["lifetime_value"] = df["monthly_charges"] * df["tenure_months"]

        return df

    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create service-related features.

        Features:
            - service_count: Total number of active services
            - has_phone_internet_bundle: Has both phone and internet
            - has_premium_support: Has tech support or online security
            - streaming_services: Number of streaming services

        Args:
            df: DataFrame with service columns

        Returns:
            DataFrame with service features
        """
        df = df.copy()

        # Service count
        service_columns = [
            "phone_service",
            "multiple_lines",
            "internet_service",
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
        ]

        # Count services (Yes = 1, No/No service = 0)
        service_count = 0
        for col in service_columns:
            if col in df.columns:
                if df[col].dtype == bool:
                    service_count += df[col].astype(int)
                else:
                    service_count += (df[col] == "Yes").astype(int)

        df["service_count"] = service_count

        # Phone + Internet bundle
        has_phone = df.get("phone_service", False)
        has_internet = df.get("internet_service", "No")

        if isinstance(has_phone, pd.Series):
            if has_phone.dtype == bool:
                phone_flag = has_phone.astype(int)
            else:
                phone_flag = (has_phone == "Yes").astype(int)
        else:
            phone_flag = 0

        if isinstance(has_internet, pd.Series):
            internet_flag = (has_internet.isin(["DSL", "Fiber optic"])).astype(int)
        else:
            internet_flag = 0

        df["has_phone_internet_bundle"] = ((phone_flag == 1) & (internet_flag == 1)).astype(int)

        # Premium support
        has_tech = df.get("tech_support", "No")
        has_security = df.get("online_security", "No")

        tech_flag = (has_tech == "Yes") if isinstance(has_tech, pd.Series) else False
        security_flag = (has_security == "Yes") if isinstance(has_security, pd.Series) else False

        df["has_premium_support"] = (tech_flag | security_flag).astype(int)

        # Streaming services count
        streaming_cols = ["streaming_tv", "streaming_movies"]
        streaming_count = 0
        for col in streaming_cols:
            if col in df.columns:
                streaming_count += (df[col] == "Yes").astype(int)
        df["streaming_services"] = streaming_count

        # Has any add-on
        addon_cols = ["online_security", "online_backup", "device_protection", "tech_support"]
        addon_count = 0
        for col in addon_cols:
            if col in df.columns:
                addon_count += (df[col] == "Yes").astype(int)
        df["addon_count"] = addon_count

        return df

    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk-related features.

        Features:
            - contract_risk_score: Risk based on contract type
            - payment_risk_score: Risk based on payment method
            - combined_risk_score: Average of contract and payment risk

        Args:
            df: DataFrame with contract and payment columns

        Returns:
            DataFrame with risk features
        """
        df = df.copy()

        # Contract risk score
        if "contract_type" in df.columns:
            contract_risk_map = {
                "Month-to-month": 1.0,
                "One year": 0.5,
                "Two year": 0.2,
            }
            df["contract_risk_score"] = df["contract_type"].map(contract_risk_map).fillna(0.5)

        # Payment risk score
        if "payment_method" in df.columns:
            payment_risk_map = {
                "Electronic check": 0.8,
                "Mailed check": 0.5,
                "Bank transfer (automatic)": 0.3,
                "Credit card (automatic)": 0.2,
            }
            df["payment_risk_score"] = df["payment_method"].map(payment_risk_map).fillna(0.5)

        # Combined risk score
        if "contract_risk_score" in df.columns and "payment_risk_score" in df.columns:
            df["combined_risk_score"] = (
                df["contract_risk_score"] + df["payment_risk_score"]
            ) / 2

        # Paperless billing risk
        if "paperless_billing" in df.columns:
            if df["paperless_billing"].dtype == bool:
                df["paperless_risk"] = df["paperless_billing"].astype(int) * 0.3
            else:
                df["paperless_risk"] = (df["paperless_billing"] == "Yes").astype(int) * 0.3

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between existing features.

        Features:
            - tenure_charge_ratio: Tenure / Monthly charges
            - value_per_service: Monthly charges / Service count
            - risk_tenure_interaction: Combined risk * Tenure risk

        Args:
            df: DataFrame with base features

        Returns:
            DataFrame with interaction features
        """
        df = df.copy()

        # Tenure-charge ratio
        if "tenure_months" in df.columns and "monthly_charges" in df.columns:
            df["tenure_charge_ratio"] = np.where(
                df["monthly_charges"] > 0,
                df["tenure_months"] / df["monthly_charges"],
                0,
            )

        # Value per service
        if "monthly_charges" in df.columns and "service_count" in df.columns:
            df["value_per_service"] = np.where(
                df["service_count"] > 0,
                df["monthly_charges"] / df["service_count"],
                df["monthly_charges"],
            )

        # Risk-tenure interaction
        if "combined_risk_score" in df.columns and "is_new_customer" in df.columns:
            df["new_customer_risk"] = df["combined_risk_score"] * df["is_new_customer"]

        # High charge + short tenure risk
        if "is_high_value" in df.columns and "is_new_customer" in df.columns:
            df["high_value_new_customer"] = df["is_high_value"] * df["is_new_customer"]

        # Senior citizen risk
        if "senior_citizen" in df.columns and "combined_risk_score" in df.columns:
            senior_flag = df["senior_citizen"]
            if senior_flag.dtype != int:
                senior_flag = senior_flag.astype(int)
            df["senior_risk_score"] = senior_flag * df["combined_risk_score"]

        return df

    def get_feature_list(self) -> List[str]:
        """
        Get list of all engineered feature names.

        Returns:
            List of feature column names
        """
        return [
            # Tenure features
            "tenure_group",
            "is_new_customer",
            "is_loyal_customer",
            "tenure_years",
            # Charge features
            "avg_monthly_charge",
            "is_high_value",
            "charge_tier",
            "lifetime_value",
            # Service features
            "service_count",
            "has_phone_internet_bundle",
            "has_premium_support",
            "streaming_services",
            "addon_count",
            # Risk features
            "contract_risk_score",
            "payment_risk_score",
            "combined_risk_score",
            "paperless_risk",
            # Interaction features
            "tenure_charge_ratio",
            "value_per_service",
            "new_customer_risk",
            "high_value_new_customer",
            "senior_risk_score",
        ]
