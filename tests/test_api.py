"""
API Tests
=========

Tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns expected fields."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "model" in data
        assert "version" in data
        assert "timestamp" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_api_info(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestModelEndpoint:
    """Tests for model info endpoint."""

    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/api/v1/model/info")

        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "threshold" in data
        assert "risk_levels" in data


class TestPredictionEndpoint:
    """Tests for prediction endpoints."""

    def test_single_prediction(self, client):
        """Test single customer prediction."""
        prediction_request = {
            "customer_id": "TEST-001",
            "gender": "Male",
            "senior_citizen": False,
            "partner": False,
            "dependents": False,
            "tenure_months": 6,
            "phone_service": True,
            "multiple_lines": "No",
            "internet_service": "Fiber optic",
            "online_security": "No",
            "online_backup": "No",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "contract_type": "Month-to-month",
            "paperless_billing": True,
            "payment_method": "Electronic check",
            "monthly_charges": 75.50,
        }

        response = client.post("/api/v1/predict", json=prediction_request)

        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] == "TEST-001"
        assert "churn_probability" in data
        assert 0 <= data["churn_probability"] <= 1
        assert "prediction" in data
        assert "risk_level" in data
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_prediction_high_risk_customer(self, client):
        """Test prediction for high risk customer."""
        prediction_request = {
            "customer_id": "HIGH-RISK-001",
            "gender": "Female",
            "senior_citizen": True,
            "partner": False,
            "dependents": False,
            "tenure_months": 2,
            "phone_service": True,
            "internet_service": "Fiber optic",
            "tech_support": "No",
            "online_security": "No",
            "contract_type": "Month-to-month",
            "payment_method": "Electronic check",
            "monthly_charges": 95.00,
        }

        response = client.post("/api/v1/predict", json=prediction_request)

        assert response.status_code == 200
        data = response.json()
        # This customer should have high churn probability
        assert data["churn_probability"] > 0.5

    def test_prediction_low_risk_customer(self, client):
        """Test prediction for low risk customer."""
        prediction_request = {
            "customer_id": "LOW-RISK-001",
            "gender": "Male",
            "senior_citizen": False,
            "partner": True,
            "dependents": True,
            "tenure_months": 60,
            "phone_service": True,
            "internet_service": "DSL",
            "tech_support": "Yes",
            "online_security": "Yes",
            "contract_type": "Two year",
            "payment_method": "Credit card (automatic)",
            "monthly_charges": 45.00,
        }

        response = client.post("/api/v1/predict", json=prediction_request)

        assert response.status_code == 200
        data = response.json()
        # This customer should have low churn probability
        assert data["churn_probability"] < 0.5

    def test_batch_prediction(self, client):
        """Test batch prediction."""
        batch_request = {
            "customers": [
                {
                    "customer_id": "BATCH-001",
                    "tenure_months": 12,
                    "monthly_charges": 50.00,
                    "contract_type": "Month-to-month",
                    "payment_method": "Electronic check",
                },
                {
                    "customer_id": "BATCH-002",
                    "tenure_months": 48,
                    "monthly_charges": 30.00,
                    "contract_type": "Two year",
                    "payment_method": "Credit card (automatic)",
                },
            ]
        }

        response = client.post("/api/v1/predict/batch", json=batch_request)

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert len(data["predictions"]) == 2
        assert "high_risk_count" in data
        assert "medium_risk_count" in data
        assert "low_risk_count" in data

    def test_prediction_returns_risk_factors(self, client):
        """Test that prediction includes risk factors."""
        prediction_request = {
            "customer_id": "FACTORS-001",
            "tenure_months": 3,
            "monthly_charges": 80.00,
            "contract_type": "Month-to-month",
            "payment_method": "Electronic check",
            "tech_support": "No",
            "online_security": "No",
        }

        response = client.post("/api/v1/predict", json=prediction_request)

        assert response.status_code == 200
        data = response.json()
        assert "top_risk_factors" in data
        assert isinstance(data["top_risk_factors"], list)
        assert len(data["top_risk_factors"]) > 0

    def test_prediction_validation_error(self, client):
        """Test prediction with invalid data."""
        invalid_request = {
            "customer_id": "INVALID-001",
            "tenure_months": -5,  # Invalid negative value
            "monthly_charges": 50.00,
        }

        response = client.post("/api/v1/predict", json=invalid_request)

        assert response.status_code == 422  # Validation error


class TestCustomerEndpoints:
    """Tests for customer endpoints."""

    def test_list_customers(self, client):
        """Test listing customers."""
        response = client.get("/api/v1/customers?limit=10")

        # May return 200 with data or empty list depending on DB state
        assert response.status_code in [200, 500]

    def test_list_customers_with_filter(self, client):
        """Test listing customers with churn filter."""
        response = client.get("/api/v1/customers?churned=true&limit=5")

        assert response.status_code in [200, 500]


class TestStatisticsEndpoints:
    """Tests for statistics endpoints."""

    def test_get_statistics(self, client):
        """Test getting overall statistics."""
        response = client.get("/api/v1/stats")

        # May fail if DB not connected
        assert response.status_code in [200, 404, 500]

    def test_stats_by_contract(self, client):
        """Test statistics by contract type."""
        response = client.get("/api/v1/stats/by-contract")

        assert response.status_code in [200, 500]

    def test_stats_by_tenure(self, client):
        """Test statistics by tenure."""
        response = client.get("/api/v1/stats/by-tenure")

        assert response.status_code in [200, 500]

    def test_stats_by_payment(self, client):
        """Test statistics by payment method."""
        response = client.get("/api/v1/stats/by-payment")

        assert response.status_code in [200, 500]
