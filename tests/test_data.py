"""
Data Module Tests
=================

Tests for data loading and preprocessing.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return DataPreprocessor()

    @pytest.fixture
    def sample_data(self):
        """Create sample customer data."""
        return pd.DataFrame(
            {
                "customer_id": ["1", "2", "3"],
                "gender": ["Male", "Female", "Male"],
                "senior_citizen": [0, 1, 0],
                "partner": ["Yes", "No", "Yes"],
                "dependents": ["No", "No", "Yes"],
                "tenure_months": [12, 24, 36],
                "phone_service": ["Yes", "Yes", "No"],
                "multiple_lines": ["No", "Yes", "No phone service"],
                "internet_service": ["DSL", "Fiber optic", "No"],
                "online_security": ["Yes", "No", "No internet service"],
                "online_backup": ["No", "Yes", "No internet service"],
                "device_protection": ["No", "Yes", "No internet service"],
                "tech_support": ["Yes", "No", "No internet service"],
                "streaming_tv": ["No", "Yes", "No internet service"],
                "streaming_movies": ["No", "Yes", "No internet service"],
                "contract_type": ["Month-to-month", "One year", "Two year"],
                "paperless_billing": ["Yes", "No", "No"],
                "payment_method": [
                    "Electronic check",
                    "Credit card (automatic)",
                    "Mailed check",
                ],
                "monthly_charges": [29.85, 89.95, 19.95],
                "total_charges": ["358.20", "2189.80", "718.20"],
                "churned": ["Yes", "No", "No"],
            }
        )

    def test_clean_data_converts_total_charges(self, preprocessor, sample_data):
        """Test that total_charges is converted to numeric."""
        cleaned = preprocessor.clean_data(sample_data)

        assert cleaned["total_charges"].dtype in [np.float64, np.float32]
        assert cleaned["total_charges"].iloc[0] == 358.20

    def test_clean_data_converts_yes_no_to_bool(self, preprocessor, sample_data):
        """Test that Yes/No columns are converted to boolean."""
        cleaned = preprocessor.clean_data(sample_data)

        assert cleaned["partner"].dtype == bool
        assert cleaned["partner"].iloc[0] is True
        assert cleaned["partner"].iloc[1] is False

    def test_clean_data_converts_churn_to_bool(self, preprocessor, sample_data):
        """Test that churned column is converted to boolean."""
        cleaned = preprocessor.clean_data(sample_data)

        assert cleaned["churned"].dtype == bool
        assert cleaned["churned"].iloc[0] is True
        assert cleaned["churned"].iloc[1] is False

    def test_handle_missing_values_median(self, preprocessor):
        """Test missing value imputation with median."""
        df = pd.DataFrame(
            {
                "numeric_col": [1.0, 2.0, np.nan, 4.0, 5.0],
                "cat_col": ["a", "b", None, "a", "b"],
            }
        )

        filled = preprocessor.handle_missing_values(df, strategy="median")

        assert filled["numeric_col"].isna().sum() == 0
        assert filled["numeric_col"].iloc[2] == 3.0  # Median of 1,2,4,5
        assert filled["cat_col"].isna().sum() == 0

    def test_encode_categorical(self, preprocessor, sample_data):
        """Test categorical encoding."""
        cleaned = preprocessor.clean_data(sample_data)
        encoded = preprocessor.encode_categorical(cleaned, fit=True)

        assert "gender_encoded" in encoded.columns
        assert "contract_type_encoded" in encoded.columns
        assert encoded["gender_encoded"].dtype in [np.int64, np.int32]

    def test_encode_categorical_transform(self, preprocessor, sample_data):
        """Test that transform uses fitted encoders."""
        cleaned = preprocessor.clean_data(sample_data)
        preprocessor.encode_categorical(cleaned, fit=True)

        new_data = pd.DataFrame({"gender": ["Female", "Male"]})
        encoded = preprocessor.encode_categorical(new_data, fit=False)

        assert "gender_encoded" in encoded.columns

    def test_scale_numeric(self, preprocessor, sample_data):
        """Test numeric scaling."""
        cleaned = preprocessor.clean_data(sample_data)
        scaled = preprocessor.scale_numeric(cleaned, fit=True)

        assert "monthly_charges_scaled" in scaled.columns
        # Scaled values should have mean ~0 and std ~1
        assert abs(scaled["monthly_charges_scaled"].mean()) < 1

    def test_fit_transform(self, preprocessor, sample_data):
        """Test full fit_transform pipeline."""
        result = preprocessor.fit_transform(sample_data)

        assert preprocessor.fitted is True
        assert "gender_encoded" in result.columns
        assert "monthly_charges_scaled" in result.columns

    def test_transform_requires_fit(self, preprocessor, sample_data):
        """Test that transform raises error if not fitted."""
        with pytest.raises(ValueError, match="Preprocessor not fitted"):
            preprocessor.transform(sample_data)

    def test_get_feature_columns(self, preprocessor):
        """Test getting feature column names."""
        features = preprocessor.get_feature_columns()

        assert isinstance(features, list)
        assert "gender_encoded" in features
        assert "monthly_charges_scaled" in features
        assert "senior_citizen" in features

    def test_prepare_for_training(self, preprocessor, sample_data):
        """Test preparing data for training."""
        processed = preprocessor.fit_transform(sample_data)
        X, y = preprocessor.prepare_for_training(processed)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert y.dtype in [np.int64, np.int32]


class TestDataLoader:
    """Tests for DataLoader class."""

    @pytest.fixture
    def loader(self, tmp_path):
        """Create loader with temp directory."""
        return DataLoader(data_dir=str(tmp_path))

    def test_init_creates_directories(self, loader):
        """Test that init sets up directory paths."""
        assert loader.raw_dir.name == "raw"
        assert loader.processed_dir.name == "processed"

    def test_load_csv_file_not_found(self, loader):
        """Test that loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            loader.load_csv("nonexistent.csv")

    def test_save_to_csv(self, loader):
        """Test saving DataFrame to CSV."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        path = loader.save_to_csv(df, "test.csv", directory="processed")

        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == 3

    def test_load_csv_after_save(self, loader):
        """Test loading previously saved CSV."""
        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        loader.save_to_csv(df, "test_load.csv", directory="processed")

        loaded = loader.load_csv("test_load.csv", directory="processed")

        assert len(loaded) == 2
        assert list(loaded.columns) == ["x", "y"]
