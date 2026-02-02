"""
API Routes
==========

FastAPI route definitions for the churn prediction API.
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChurnStatistics,
    ContractStats,
    CustomerDetail,
    HealthCheck,
    HighRiskCustomer,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)
from src.config import settings
from src.database import check_connection, get_db

# Create routers
router = APIRouter()
customers_router = APIRouter(prefix="/customers", tags=["customers"])
predictions_router = APIRouter(prefix="/predict", tags=["predictions"])
model_router = APIRouter(prefix="/model", tags=["model"])
stats_router = APIRouter(prefix="/stats", tags=["statistics"])


# =====================================================
# Health Check
# =====================================================


@router.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """
    Check API health status.

    Returns database connection status and model availability.
    """
    db_status = "healthy" if check_connection() else "unhealthy"

    # Check model status (simplified for now)
    model_status = "loaded"  # Would check actual model in production

    return HealthCheck(
        status="healthy" if db_status == "healthy" else "degraded",
        database=db_status,
        model=model_status,
        version=settings.api_version,
        timestamp=datetime.now(),
    )


# =====================================================
# Customer Routes
# =====================================================


@customers_router.get("", response_model=List[CustomerDetail])
async def list_customers(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    churned: Optional[bool] = Query(None, description="Filter by churn status"),
    db: Session = Depends(get_db),
):
    """
    List all customers with optional filtering.

    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return
    - **churned**: Filter by churn status (true/false)
    """
    query = "SELECT * FROM v_customer_360"
    params = {}

    if churned is not None:
        query += " WHERE churned = :churned"
        params["churned"] = churned

    query += f" LIMIT {limit} OFFSET {skip}"

    result = db.execute(text(query), params)
    columns = result.keys()
    rows = result.fetchall()

    return [dict(zip(columns, row)) for row in rows]


@customers_router.get("/{customer_id}", response_model=CustomerDetail)
async def get_customer(
    customer_id: str,
    db: Session = Depends(get_db),
):
    """
    Get detailed information for a specific customer.

    - **customer_id**: Unique customer identifier
    """
    query = "SELECT * FROM v_customer_360 WHERE customer_id = :customer_id"
    result = db.execute(text(query), {"customer_id": customer_id})
    row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    columns = result.keys()
    return dict(zip(columns, row))


@customers_router.get("/high-risk", response_model=List[HighRiskCustomer])
async def get_high_risk_customers(
    limit: int = Query(10, ge=1, le=100, description="Max records to return"),
    db: Session = Depends(get_db),
):
    """
    Get customers with highest churn risk.

    Returns customers that haven't churned but have high risk scores.
    """
    query = f"SELECT * FROM v_high_risk_customers LIMIT {limit}"
    result = db.execute(text(query))
    columns = result.keys()
    rows = result.fetchall()

    return [dict(zip(columns, row)) for row in rows]


# =====================================================
# Prediction Routes
# =====================================================


@predictions_router.post("", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    db: Session = Depends(get_db),
):
    """
    Predict churn probability for a single customer.

    Accepts customer data and returns churn probability with risk factors.
    """
    # Convert request to dict
    customer_data = request.model_dump()

    # Calculate total_charges if not provided
    if customer_data.get("total_charges") is None:
        customer_data["total_charges"] = (
            customer_data["monthly_charges"] * customer_data["tenure_months"]
        )

    # Create a simple rule-based prediction for demo
    # In production, this would use the trained ML model
    risk_score = calculate_risk_score(customer_data)
    risk_factors = identify_risk_factors(customer_data)

    response = PredictionResponse(
        customer_id=customer_data["customer_id"],
        churn_probability=risk_score,
        prediction=risk_score >= settings.churn_probability_threshold,
        risk_level=settings.get_risk_level(risk_score),
        top_risk_factors=risk_factors[:5],
        model_version=settings.model_version,
        predicted_at=datetime.now(),
    )

    # Store prediction in database
    store_prediction(db, response)

    return response


@predictions_router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    db: Session = Depends(get_db),
):
    """
    Predict churn for multiple customers.

    Accepts a list of customers and returns predictions for all.
    """
    predictions = []
    high_risk = 0
    medium_risk = 0
    low_risk = 0

    for customer in request.customers:
        customer_data = customer.model_dump()

        if customer_data.get("total_charges") is None:
            customer_data["total_charges"] = (
                customer_data["monthly_charges"] * customer_data["tenure_months"]
            )

        risk_score = calculate_risk_score(customer_data)
        risk_factors = identify_risk_factors(customer_data)
        risk_level = settings.get_risk_level(risk_score)

        if risk_level == "HIGH":
            high_risk += 1
        elif risk_level == "MEDIUM":
            medium_risk += 1
        else:
            low_risk += 1

        predictions.append(
            PredictionResponse(
                customer_id=customer_data["customer_id"],
                churn_probability=risk_score,
                prediction=risk_score >= settings.churn_probability_threshold,
                risk_level=risk_level,
                top_risk_factors=risk_factors[:5],
                model_version=settings.model_version,
                predicted_at=datetime.now(),
            )
        )

    return BatchPredictionResponse(
        predictions=predictions,
        total_count=len(predictions),
        high_risk_count=high_risk,
        medium_risk_count=medium_risk,
        low_risk_count=low_risk,
    )


# =====================================================
# Model Routes
# =====================================================


@model_router.get("/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the current prediction model.
    """
    return ModelInfo(
        model_name=settings.model_name,
        model_version=settings.model_version,
        model_type="XGBoost",  # Would be dynamic in production
        threshold=settings.churn_probability_threshold,
        risk_levels={
            "high": f">= {settings.high_risk_threshold}",
            "medium": f">= {settings.medium_risk_threshold}",
            "low": f"< {settings.medium_risk_threshold}",
        },
        is_active=True,
    )


# =====================================================
# Statistics Routes
# =====================================================


@stats_router.get("", response_model=ChurnStatistics)
async def get_churn_statistics(
    db: Session = Depends(get_db),
):
    """
    Get overall churn statistics.
    """
    query = "SELECT * FROM v_churn_summary"
    result = db.execute(text(query))
    row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="No statistics available")

    columns = result.keys()
    return dict(zip(columns, row))


@stats_router.get("/by-contract", response_model=List[ContractStats])
async def get_stats_by_contract(
    db: Session = Depends(get_db),
):
    """
    Get churn statistics grouped by contract type.
    """
    query = "SELECT * FROM v_churn_by_contract"
    result = db.execute(text(query))
    columns = result.keys()
    rows = result.fetchall()

    return [dict(zip(columns, row)) for row in rows]


@stats_router.get("/by-tenure")
async def get_stats_by_tenure(
    db: Session = Depends(get_db),
):
    """
    Get churn statistics grouped by tenure.
    """
    query = "SELECT * FROM v_churn_by_tenure"
    result = db.execute(text(query))
    columns = result.keys()
    rows = result.fetchall()

    return [dict(zip(columns, row)) for row in rows]


@stats_router.get("/by-payment")
async def get_stats_by_payment(
    db: Session = Depends(get_db),
):
    """
    Get churn statistics grouped by payment method.
    """
    query = "SELECT * FROM v_churn_by_payment"
    result = db.execute(text(query))
    columns = result.keys()
    rows = result.fetchall()

    return [dict(zip(columns, row)) for row in rows]


# =====================================================
# Helper Functions
# =====================================================


def calculate_risk_score(data: dict) -> float:
    """
    Calculate churn risk score using rules.

    This is a simplified version. In production, use ML model.
    """
    score = 0.3  # Base score

    # Contract type
    if data.get("contract_type") == "Month-to-month":
        score += 0.25
    elif data.get("contract_type") == "One year":
        score += 0.1

    # Tenure
    tenure = data.get("tenure_months", 0)
    if tenure <= 6:
        score += 0.2
    elif tenure <= 12:
        score += 0.1
    elif tenure >= 48:
        score -= 0.15

    # Payment method
    if data.get("payment_method") == "Electronic check":
        score += 0.15

    # Support services
    if data.get("tech_support") != "Yes" and data.get("online_security") != "Yes":
        score += 0.1

    # Internet service
    if data.get("internet_service") == "Fiber optic":
        score += 0.1

    # Monthly charges
    if data.get("monthly_charges", 0) > 70:
        score += 0.05

    # Senior citizen
    if data.get("senior_citizen"):
        score += 0.05

    return min(max(score, 0), 1)  # Clamp between 0 and 1


def identify_risk_factors(data: dict) -> List[str]:
    """Identify risk factors for a customer."""
    factors = []

    if data.get("contract_type") == "Month-to-month":
        factors.append("Month-to-month contract (high churn risk)")

    tenure = data.get("tenure_months", 0)
    if tenure <= 12:
        factors.append(f"Short tenure ({tenure} months)")

    if data.get("tech_support") != "Yes" and data.get("online_security") != "Yes":
        factors.append("No premium support services")

    if data.get("payment_method") == "Electronic check":
        factors.append("Electronic check payment method")

    if data.get("monthly_charges", 0) > 70:
        factors.append(f"High monthly charges (${data.get('monthly_charges', 0):.2f})")

    if data.get("internet_service") == "Fiber optic":
        factors.append("Fiber optic internet")

    if data.get("senior_citizen"):
        factors.append("Senior citizen segment")

    if not data.get("partner") and not data.get("dependents"):
        factors.append("No partner or dependents")

    return factors


def store_prediction(db: Session, prediction: PredictionResponse) -> None:
    """Store prediction in database."""
    try:
        query = """
            INSERT INTO predictions (
                customer_id, churn_probability, prediction,
                risk_level, risk_factors, model_version, predicted_at
            ) VALUES (
                :customer_id, :probability, :prediction,
                :risk_level, :risk_factors, :model_version, :predicted_at
            )
        """
        db.execute(
            text(query),
            {
                "customer_id": prediction.customer_id,
                "probability": prediction.churn_probability,
                "prediction": prediction.prediction,
                "risk_level": prediction.risk_level,
                "risk_factors": str(prediction.top_risk_factors),
                "model_version": prediction.model_version,
                "predicted_at": prediction.predicted_at,
            },
        )
        db.commit()
    except Exception:
        db.rollback()
