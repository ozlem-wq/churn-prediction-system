"""
API Schemas
===========

Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =====================================================
# Customer Schemas
# =====================================================


class CustomerBase(BaseModel):
    """Base customer schema."""

    customer_id: str = Field(..., description="Unique customer identifier")
    gender: Optional[str] = Field(None, description="Customer gender (Male/Female)")
    senior_citizen: Optional[bool] = Field(False, description="Is senior citizen")
    partner: Optional[bool] = Field(False, description="Has partner")
    dependents: Optional[bool] = Field(False, description="Has dependents")
    tenure_months: Optional[int] = Field(0, ge=0, description="Months as customer")


class CustomerCreate(CustomerBase):
    """Schema for creating a customer."""

    pass


class CustomerResponse(CustomerBase):
    """Schema for customer response."""

    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class CustomerDetail(CustomerBase):
    """Detailed customer response with all related data."""

    # Services
    phone_service: Optional[bool] = None
    multiple_lines: Optional[str] = None
    internet_service: Optional[str] = None
    online_security: Optional[str] = None
    online_backup: Optional[str] = None
    device_protection: Optional[str] = None
    tech_support: Optional[str] = None
    streaming_tv: Optional[str] = None
    streaming_movies: Optional[str] = None

    # Billing
    contract_type: Optional[str] = None
    paperless_billing: Optional[bool] = None
    payment_method: Optional[str] = None
    monthly_charges: Optional[float] = None
    total_charges: Optional[float] = None

    # Churn
    churned: Optional[bool] = None
    churn_date: Optional[datetime] = None


# =====================================================
# Prediction Schemas
# =====================================================


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""

    customer_id: str = Field(..., description="Customer identifier")
    gender: Optional[str] = Field("Male", description="Gender (Male/Female)")
    senior_citizen: Optional[bool] = Field(False, description="Is senior citizen")
    partner: Optional[bool] = Field(False, description="Has partner")
    dependents: Optional[bool] = Field(False, description="Has dependents")
    tenure_months: int = Field(..., ge=0, description="Months as customer")

    # Services
    phone_service: Optional[bool] = Field(True, description="Has phone service")
    multiple_lines: Optional[str] = Field("No", description="Multiple lines")
    internet_service: Optional[str] = Field("DSL", description="Internet type")
    online_security: Optional[str] = Field("No", description="Online security")
    online_backup: Optional[str] = Field("No", description="Online backup")
    device_protection: Optional[str] = Field("No", description="Device protection")
    tech_support: Optional[str] = Field("No", description="Tech support")
    streaming_tv: Optional[str] = Field("No", description="Streaming TV")
    streaming_movies: Optional[str] = Field("No", description="Streaming movies")

    # Billing
    contract_type: Optional[str] = Field("Month-to-month", description="Contract type")
    paperless_billing: Optional[bool] = Field(True, description="Paperless billing")
    payment_method: Optional[str] = Field("Electronic check", description="Payment method")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges")
    total_charges: Optional[float] = Field(None, ge=0, description="Total charges")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "7590-VHVEG",
                "gender": "Female",
                "senior_citizen": False,
                "tenure_months": 12,
                "monthly_charges": 29.85,
                "contract_type": "Month-to-month",
                "internet_service": "DSL",
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction."""

    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    prediction: bool = Field(..., description="Churn prediction (True/False)")
    risk_level: str = Field(..., description="Risk level (LOW/MEDIUM/HIGH)")
    top_risk_factors: Optional[List[str]] = Field(None, description="Top risk factors")
    model_version: Optional[str] = Field(None, description="Model version used")
    predicted_at: Optional[datetime] = Field(None, description="Prediction timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "7590-VHVEG",
                "churn_probability": 0.73,
                "prediction": True,
                "risk_level": "HIGH",
                "top_risk_factors": [
                    "Month-to-month contract",
                    "Low tenure (12 months)",
                    "No tech support",
                ],
                "model_version": "v1.0.0",
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""

    customers: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""

    predictions: List[PredictionResponse]
    total_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


# =====================================================
# Model Schemas
# =====================================================


class ModelInfo(BaseModel):
    """Model information schema."""

    model_name: str
    model_version: str
    model_type: str
    threshold: float
    risk_levels: Dict[str, str]
    is_active: bool = True


class ModelMetrics(BaseModel):
    """Model metrics schema."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_date: Optional[datetime] = None


# =====================================================
# Statistics Schemas
# =====================================================


class ChurnStatistics(BaseModel):
    """Churn statistics schema."""

    total_customers: int
    churned_customers: int
    active_customers: int
    churn_rate: float
    avg_monthly_charges: float
    avg_total_charges: float
    avg_tenure_months: float


class ContractStats(BaseModel):
    """Statistics by contract type."""

    contract_type: str
    customer_count: int
    churned_count: int
    churn_rate: float
    avg_monthly_charges: float


class HighRiskCustomer(BaseModel):
    """High risk customer schema."""

    customer_id: str
    tenure_months: int
    contract_type: str
    monthly_charges: float
    combined_risk_score: float


# =====================================================
# Health Check Schemas
# =====================================================


class HealthCheck(BaseModel):
    """Health check response."""

    status: str
    database: str
    model: str
    version: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
