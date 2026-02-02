"""
Feature Engineering Tests
=========================

Tests for feature engineering module.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import FeatureEngineer


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    @pytest.fixture
    def engineer(self):
        """Create feature engineer instance."""
        return FeatureEngineer()

    @pytest.fixture
    def sample_data(self):
        """Create sample customer data."""
        return pd.DataFrame(
            {
                "customer_id": ["1", "2", "3", "4"],
                "tenure_months": [3, 15, 30, 60],
                "monthly_charges": [29.85, 55.00, 89.95, 45.00],
                "total_charges": [89.55, 825.00, 2698.50, 2700.00],
                "contract_type": [
                    "Month-to-month",
                    "One year",
                    "Two year",
                    "Month-to-month",
                ],
                "payment_method": [
                    "Electronic check",
                    "Credit card (automatic)",
                    "Bank transfer (automatic)",
                    "Mailed check",
                ],
                "phone_service": [True, True, False, True],
                "internet_service": ["DSL", "Fiber optic", "No", "DSL"],
                "tech_support": ["No", "Yes", "No internet service", "No"],
                "online_security": ["Yes", "No", "No internet service", "No"],
                "streaming_tv": ["No", "Yes", "No internet service", "Yes"],
                "streaming_movies": ["No", "Yes", "No internet service", "Yes"],
                "senior_citizen": [False, True, False, False],
                "partner": [True, False, True, False],
                "dependents": [False, False, True, False],
                "paperless_billing": ["Yes", "No", "No", "Yes"],
            }
        )

    def test_create_tenure_features(self, engineer, sample_data):
        """Test tenure feature creation."""
        result = engineer.create_tenure_features(sample_data)

        assert "tenure_group" in result.columns
        assert "is_new_customer" in result.columns
        assert "is_loyal_customer" in result.columns
        assert "tenure_years" in result.columns

        # Check new customer flag
        assert result.loc[0, "is_new_customer"] == 1  # 3 months
        assert result.loc[2, "is_new_customer"] == 0  # 30 months

        # Check loyal customer flag
        assert result.loc[3, "is_loyal_customer"] == 1  # 60 months
        assert result.loc[0, "is_loyal_customer"] == 0  # 3 months

    def test_create_charge_features(self, engineer, sample_data):
        """Test charge feature creation."""
        result = engineer.create_charge_features(sample_data)

        assert "avg_monthly_charge" in result.columns
        assert "is_high_value" in result.columns
        assert "charge_tier" in result.columns
        assert "lifetime_value" in result.columns

        # Check avg monthly charge calculation
        expected_avg = sample_data["total_charges"] / sample_data["tenure_months"]
        np.testing.assert_array_almost_equal(
            result["avg_monthly_charge"].values, expected_avg.values, decimal=2
        )

        # Check high value flag (monthly > 70)
        assert result.loc[2, "is_high_value"] == 1  # 89.95
        assert result.loc[0, "is_high_value"] == 0  # 29.85

    def test_create_service_features(self, engineer, sample_data):
        """Test service feature creation."""
        result = engineer.create_service_features(sample_data)

        assert "service_count" in result.columns
        assert "has_phone_internet_bundle" in result.columns
        assert "has_premium_support" in result.columns
        assert "streaming_services" in result.columns
        assert "addon_count" in result.columns

        # Customer 1: phone=Yes, internet=Fiber, tv=Yes, movies=Yes, security=No, backup=No
        # Should have streaming_services = 2
        assert result.loc[1, "streaming_services"] == 2

        # Customer 0: has online_security=Yes, so has_premium_support should be 1
        assert result.loc[0, "has_premium_support"] == 1

    def test_create_risk_features(self, engineer, sample_data):
        """Test risk feature creation."""
        result = engineer.create_risk_features(sample_data)

        assert "contract_risk_score" in result.columns
        assert "payment_risk_score" in result.columns
        assert "combined_risk_score" in result.columns

        # Month-to-month contract should have highest risk
        assert result.loc[0, "contract_risk_score"] == 1.0
        # Two year contract should have lowest risk
        assert result.loc[2, "contract_risk_score"] == 0.2

        # Electronic check should have high payment risk
        assert result.loc[0, "payment_risk_score"] == 0.8
        # Credit card automatic should have low payment risk
        assert result.loc[1, "payment_risk_score"] == 0.2

    def test_create_interaction_features(self, engineer, sample_data):
        """Test interaction feature creation."""
        # First create base features
        result = engineer.create_tenure_features(sample_data)
        result = engineer.create_charge_features(result)
        result = engineer.create_service_features(result)
        result = engineer.create_risk_features(result)
        result = engineer.create_interaction_features(result)

        assert "tenure_charge_ratio" in result.columns
        assert "value_per_service" in result.columns
        assert "new_customer_risk" in result.columns

    def test_create_all_features(self, engineer, sample_data):
        """Test creating all features at once."""
        result = engineer.create_all_features(sample_data)

        # Check all feature categories are present
        assert "tenure_group" in result.columns
        assert "avg_monthly_charge" in result.columns
        assert "service_count" in result.columns
        assert "contract_risk_score" in result.columns
        assert "tenure_charge_ratio" in result.columns

    def test_get_feature_list(self, engineer):
        """Test getting list of feature names."""
        features = engineer.get_feature_list()

        assert isinstance(features, list)
        assert len(features) > 0
        assert "tenure_group" in features
        assert "service_count" in features
        assert "combined_risk_score" in features

    def test_handles_missing_columns(self, engineer):
        """Test that missing columns don't cause errors."""
        df = pd.DataFrame(
            {
                "customer_id": ["1", "2"],
                "tenure_months": [12, 24],
            }
        )

        result = engineer.create_tenure_features(df)

        assert "tenure_group" in result.columns
        assert len(result) == 2

    def test_handles_zero_tenure(self, engineer):
        """Test handling of zero tenure values."""
        df = pd.DataFrame(
            {
                "customer_id": ["1"],
                "tenure_months": [0],
                "monthly_charges": [50.0],
                "total_charges": [0.0],
            }
        )

        result = engineer.create_charge_features(df)

        # avg_monthly_charge should use monthly_charges when tenure is 0
        assert result["avg_monthly_charge"].iloc[0] == 50.0
