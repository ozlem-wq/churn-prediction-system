-- Churn Prediction System - Database Schema
-- Version: 1.0.0

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- Müşteri Ana Tablosu
-- =====================================================
CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) UNIQUE NOT NULL,
    gender VARCHAR(10),
    senior_citizen BOOLEAN DEFAULT FALSE,
    partner BOOLEAN DEFAULT FALSE,
    dependents BOOLEAN DEFAULT FALSE,
    tenure_months INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_customers_customer_id ON customers(customer_id);
CREATE INDEX IF NOT EXISTS idx_customers_tenure ON customers(tenure_months);

-- =====================================================
-- Servis Bilgileri Tablosu
-- =====================================================
CREATE TABLE IF NOT EXISTS services (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    phone_service BOOLEAN DEFAULT FALSE,
    multiple_lines VARCHAR(20) DEFAULT 'No',
    internet_service VARCHAR(20) DEFAULT 'No',
    online_security VARCHAR(20) DEFAULT 'No',
    online_backup VARCHAR(20) DEFAULT 'No',
    device_protection VARCHAR(20) DEFAULT 'No',
    tech_support VARCHAR(20) DEFAULT 'No',
    streaming_tv VARCHAR(20) DEFAULT 'No',
    streaming_movies VARCHAR(20) DEFAULT 'No',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(customer_id)
);

CREATE INDEX IF NOT EXISTS idx_services_customer_id ON services(customer_id);
CREATE INDEX IF NOT EXISTS idx_services_internet ON services(internet_service);

-- =====================================================
-- Ödeme Bilgileri Tablosu
-- =====================================================
CREATE TABLE IF NOT EXISTS billing (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    contract_type VARCHAR(20) DEFAULT 'Month-to-month',
    paperless_billing BOOLEAN DEFAULT FALSE,
    payment_method VARCHAR(50) DEFAULT 'Electronic check',
    monthly_charges DECIMAL(10,2) DEFAULT 0.00,
    total_charges DECIMAL(10,2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(customer_id)
);

CREATE INDEX IF NOT EXISTS idx_billing_customer_id ON billing(customer_id);
CREATE INDEX IF NOT EXISTS idx_billing_contract ON billing(contract_type);
CREATE INDEX IF NOT EXISTS idx_billing_monthly_charges ON billing(monthly_charges);

-- =====================================================
-- Churn Etiketleri Tablosu
-- =====================================================
CREATE TABLE IF NOT EXISTS churn_labels (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    churned BOOLEAN DEFAULT FALSE,
    churn_date DATE,
    churn_reason VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(customer_id)
);

CREATE INDEX IF NOT EXISTS idx_churn_customer_id ON churn_labels(customer_id);
CREATE INDEX IF NOT EXISTS idx_churn_churned ON churn_labels(churned);

-- =====================================================
-- Model Tahminleri Tablosu
-- =====================================================
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    churn_probability FLOAT NOT NULL,
    prediction BOOLEAN NOT NULL,
    risk_level VARCHAR(20),
    risk_factors JSONB,
    model_version VARCHAR(50) NOT NULL,
    model_name VARCHAR(100),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_customer_id ON predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at ON predictions(predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON predictions(risk_level);

-- =====================================================
-- Model Metadata Tablosu (MLflow alternatifi olarak)
-- =====================================================
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_path VARCHAR(255) NOT NULL,
    algorithm VARCHAR(50),
    accuracy FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    roc_auc FLOAT,
    training_date TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE,
    hyperparameters JSONB,
    feature_importance JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_model_active ON model_registry(is_active);

-- =====================================================
-- Feature Store Tablosu
-- =====================================================
CREATE TABLE IF NOT EXISTS feature_store (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    avg_monthly_charge FLOAT,
    tenure_group VARCHAR(20),
    service_count INTEGER,
    has_premium_support BOOLEAN,
    contract_risk_score FLOAT,
    payment_risk_score FLOAT,
    total_services_value FLOAT,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(customer_id)
);

CREATE INDEX IF NOT EXISTS idx_feature_store_customer_id ON feature_store(customer_id);

-- =====================================================
-- Audit Log Tablosu
-- =====================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    operation VARCHAR(10) NOT NULL,
    record_id INTEGER,
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(100),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_table ON audit_log(table_name);
CREATE INDEX IF NOT EXISTS idx_audit_changed_at ON audit_log(changed_at DESC);

-- =====================================================
-- Updated_at Trigger Function
-- =====================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables with updated_at column
CREATE TRIGGER update_customers_updated_at
    BEFORE UPDATE ON customers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_services_updated_at
    BEFORE UPDATE ON services
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_billing_updated_at
    BEFORE UPDATE ON billing
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_churn_labels_updated_at
    BEFORE UPDATE ON churn_labels
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
