"""
Configuration Management
========================

Centralized configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/churn_db"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "churn_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = True
    api_title: str = "Churn Prediction API"
    api_version: str = "1.0.0"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "churn-prediction"

    # Model
    model_path: str = "models/"
    model_version: str = "v1.0.0"
    model_name: str = "churn_classifier"

    # Feature Engineering
    tenure_bins: list = [0, 12, 24, 48, 72]
    tenure_labels: list = ["0-12", "13-24", "25-48", "49+"]

    # Training
    test_size: float = 0.2
    validation_size: float = 0.15
    random_state: int = 42
    cv_folds: int = 5

    # Thresholds
    churn_probability_threshold: float = 0.5
    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.4

    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL."""
        return self.database_url

    @property
    def database_url_async(self) -> str:
        """Get async database URL for asyncpg."""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")

    def get_risk_level(self, probability: float) -> str:
        """Determine risk level based on churn probability."""
        if probability >= self.high_risk_threshold:
            return "HIGH"
        elif probability >= self.medium_risk_threshold:
            return "MEDIUM"
        else:
            return "LOW"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
