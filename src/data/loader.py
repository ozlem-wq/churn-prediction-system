"""
Data Loader
===========

Handles loading data from various sources:
- CSV files
- PostgreSQL database
- Feature store
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sqlalchemy import text

from src.config import settings
from src.database import get_db_context


class DataLoader:
    """Load data from various sources."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader.

        Args:
            data_dir: Base directory for data files
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

    def load_csv(
        self,
        filename: str,
        directory: str = "raw",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filename: Name of CSV file
            directory: Subdirectory ("raw" or "processed")
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame with loaded data
        """
        if directory == "raw":
            filepath = self.raw_dir / filename
        else:
            filepath = self.processed_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        return pd.read_csv(filepath, **kwargs)

    def load_telco_churn(self, filename: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
        """
        Load Telco Customer Churn dataset.

        This is the standard dataset format from Kaggle.

        Args:
            filename: Name of the Telco churn CSV file

        Returns:
            DataFrame with Telco churn data
        """
        df = self.load_csv(filename)

        # Standard column renaming
        column_mapping = {
            "customerID": "customer_id",
            "SeniorCitizen": "senior_citizen",
            "Partner": "partner",
            "Dependents": "dependents",
            "tenure": "tenure_months",
            "PhoneService": "phone_service",
            "MultipleLines": "multiple_lines",
            "InternetService": "internet_service",
            "OnlineSecurity": "online_security",
            "OnlineBackup": "online_backup",
            "DeviceProtection": "device_protection",
            "TechSupport": "tech_support",
            "StreamingTV": "streaming_tv",
            "StreamingMovies": "streaming_movies",
            "Contract": "contract_type",
            "PaperlessBilling": "paperless_billing",
            "PaymentMethod": "payment_method",
            "MonthlyCharges": "monthly_charges",
            "TotalCharges": "total_charges",
            "Churn": "churned",
        }

        # Rename columns if they exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        return df

    def load_from_db(
        self,
        query: Optional[str] = None,
        table: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load data from PostgreSQL database.

        Args:
            query: SQL query to execute
            table: Table name (alternative to query)

        Returns:
            DataFrame with query results
        """
        if query is None and table is None:
            raise ValueError("Either query or table must be provided")

        if query is None:
            query = f"SELECT * FROM {table}"

        with get_db_context() as db:
            result = db.execute(text(query))
            columns = result.keys()
            data = result.fetchall()

        return pd.DataFrame(data, columns=columns)

    def load_customer_360(self) -> pd.DataFrame:
        """
        Load complete customer view from database.

        Returns:
            DataFrame with all customer data joined
        """
        return self.load_from_db(query="SELECT * FROM v_customer_360")

    def load_ml_features(self) -> pd.DataFrame:
        """
        Load ML-ready feature matrix from database.

        Returns:
            DataFrame with encoded features ready for training
        """
        return self.load_from_db(query="SELECT * FROM v_ml_features")

    def load_high_risk_customers(self) -> pd.DataFrame:
        """
        Load high-risk customers from database.

        Returns:
            DataFrame with high-risk customers
        """
        return self.load_from_db(query="SELECT * FROM v_high_risk_customers")

    def save_to_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        directory: str = "processed",
        **kwargs,
    ) -> Path:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save
            filename: Output filename
            directory: Subdirectory ("raw" or "processed")
            **kwargs: Additional arguments for df.to_csv

        Returns:
            Path to saved file
        """
        if directory == "raw":
            output_dir = self.raw_dir
        else:
            output_dir = self.processed_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename

        df.to_csv(filepath, index=False, **kwargs)
        return filepath

    def insert_to_db(
        self,
        df: pd.DataFrame,
        table: str,
        if_exists: str = "append",
    ) -> int:
        """
        Insert DataFrame to database table.

        Args:
            df: DataFrame to insert
            table: Target table name
            if_exists: How to handle existing table ('fail', 'replace', 'append')

        Returns:
            Number of rows inserted
        """
        from sqlalchemy import create_engine

        engine = create_engine(settings.database_url_sync)
        rows = df.to_sql(table, engine, if_exists=if_exists, index=False)
        return rows or len(df)
