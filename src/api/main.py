"""
FastAPI Main Application
========================

Entry point for the Churn Prediction API.
"""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import (
    customers_router,
    model_router,
    predictions_router,
    router,
    stats_router,
)
from src.config import settings
from src.database import check_connection


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Runs on startup and shutdown.
    """
    # Startup
    print("Starting Churn Prediction API...")

    # Check database connection
    if check_connection():
        print("Database connection: OK")
    else:
        print("Database connection: FAILED")

    yield

    # Shutdown
    print("Shutting down Churn Prediction API...")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description="""
    ## Churn Prediction API

    A machine learning powered API for predicting customer churn.

    ### Features
    - **Single Prediction**: Predict churn for individual customers
    - **Batch Prediction**: Predict churn for multiple customers at once
    - **Customer Data**: Access customer information and history
    - **Statistics**: View churn analytics and trends
    - **Model Info**: Get information about the prediction model

    ### Risk Levels
    - **HIGH**: Probability >= 0.7 (Immediate attention required)
    - **MEDIUM**: Probability >= 0.4 (Monitor closely)
    - **LOW**: Probability < 0.4 (Low churn risk)
    """,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.api_debug else "An error occurred",
            "timestamp": datetime.now().isoformat(),
        },
    )


# Include routers
app.include_router(router)
app.include_router(customers_router, prefix="/api/v1")
app.include_router(predictions_router, prefix="/api/v1")
app.include_router(model_router, prefix="/api/v1")
app.include_router(stats_router, prefix="/api/v1")


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    API root endpoint.

    Returns basic API information.
    """
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "customers": "/api/v1/customers",
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/predict/batch",
            "model_info": "/api/v1/model/info",
            "statistics": "/api/v1/stats",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
    )
