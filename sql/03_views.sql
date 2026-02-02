-- Churn Prediction System - Analytical Views
-- Version: 1.0.0

-- =====================================================
-- Tam Müşteri Görünümü (Complete Customer View)
-- Tüm tabloları birleştiren ana görünüm
-- =====================================================
CREATE OR REPLACE VIEW v_customer_360 AS
SELECT
    c.customer_id,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    c.tenure_months,

    -- Services
    s.phone_service,
    s.multiple_lines,
    s.internet_service,
    s.online_security,
    s.online_backup,
    s.device_protection,
    s.tech_support,
    s.streaming_tv,
    s.streaming_movies,

    -- Billing
    b.contract_type,
    b.paperless_billing,
    b.payment_method,
    b.monthly_charges,
    b.total_charges,

    -- Churn
    COALESCE(cl.churned, false) as churned,
    cl.churn_date,
    cl.churn_reason,

    -- Computed fields
    CASE
        WHEN c.tenure_months <= 12 THEN '0-12 months'
        WHEN c.tenure_months <= 24 THEN '13-24 months'
        WHEN c.tenure_months <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END as tenure_group,

    CASE
        WHEN c.tenure_months > 0 THEN ROUND(b.total_charges / c.tenure_months, 2)
        ELSE b.monthly_charges
    END as avg_monthly_charge,

    -- Service count
    (
        CASE WHEN s.phone_service THEN 1 ELSE 0 END +
        CASE WHEN s.internet_service != 'No' THEN 1 ELSE 0 END +
        CASE WHEN s.online_security = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN s.online_backup = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN s.device_protection = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN s.tech_support = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN s.streaming_tv = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN s.streaming_movies = 'Yes' THEN 1 ELSE 0 END
    ) as service_count,

    -- Premium support flag
    (s.tech_support = 'Yes' OR s.online_security = 'Yes') as has_premium_support,

    -- Contract risk (month-to-month = high risk)
    CASE b.contract_type
        WHEN 'Month-to-month' THEN 1.0
        WHEN 'One year' THEN 0.5
        WHEN 'Two year' THEN 0.2
        ELSE 0.5
    END as contract_risk_score,

    -- Payment risk (electronic check = higher risk)
    CASE b.payment_method
        WHEN 'Electronic check' THEN 0.8
        WHEN 'Mailed check' THEN 0.5
        WHEN 'Bank transfer (automatic)' THEN 0.3
        WHEN 'Credit card (automatic)' THEN 0.2
        ELSE 0.5
    END as payment_risk_score

FROM customers c
LEFT JOIN services s ON c.customer_id = s.customer_id
LEFT JOIN billing b ON c.customer_id = b.customer_id
LEFT JOIN churn_labels cl ON c.customer_id = cl.customer_id;

-- =====================================================
-- Churn Özet İstatistikleri
-- =====================================================
CREATE OR REPLACE VIEW v_churn_summary AS
SELECT
    COUNT(*) as total_customers,
    SUM(CASE WHEN churned THEN 1 ELSE 0 END) as churned_customers,
    SUM(CASE WHEN NOT churned THEN 1 ELSE 0 END) as active_customers,
    ROUND(100.0 * SUM(CASE WHEN churned THEN 1 ELSE 0 END) / COUNT(*), 2) as churn_rate,
    ROUND(AVG(monthly_charges), 2) as avg_monthly_charges,
    ROUND(AVG(total_charges), 2) as avg_total_charges,
    ROUND(AVG(tenure_months), 1) as avg_tenure_months
FROM v_customer_360;

-- =====================================================
-- Sözleşme Tipine Göre Churn Analizi
-- =====================================================
CREATE OR REPLACE VIEW v_churn_by_contract AS
SELECT
    contract_type,
    COUNT(*) as customer_count,
    SUM(CASE WHEN churned THEN 1 ELSE 0 END) as churned_count,
    ROUND(100.0 * SUM(CASE WHEN churned THEN 1 ELSE 0 END) / COUNT(*), 2) as churn_rate,
    ROUND(AVG(monthly_charges), 2) as avg_monthly_charges,
    ROUND(AVG(tenure_months), 1) as avg_tenure
FROM v_customer_360
GROUP BY contract_type
ORDER BY churn_rate DESC;

-- =====================================================
-- İnternet Servisine Göre Churn Analizi
-- =====================================================
CREATE OR REPLACE VIEW v_churn_by_internet AS
SELECT
    internet_service,
    COUNT(*) as customer_count,
    SUM(CASE WHEN churned THEN 1 ELSE 0 END) as churned_count,
    ROUND(100.0 * SUM(CASE WHEN churned THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) as churn_rate,
    ROUND(AVG(monthly_charges), 2) as avg_monthly_charges
FROM v_customer_360
GROUP BY internet_service
ORDER BY churn_rate DESC;

-- =====================================================
-- Tenure Grubuna Göre Churn Analizi
-- =====================================================
CREATE OR REPLACE VIEW v_churn_by_tenure AS
SELECT
    tenure_group,
    COUNT(*) as customer_count,
    SUM(CASE WHEN churned THEN 1 ELSE 0 END) as churned_count,
    ROUND(100.0 * SUM(CASE WHEN churned THEN 1 ELSE 0 END) / COUNT(*), 2) as churn_rate,
    ROUND(AVG(monthly_charges), 2) as avg_monthly_charges
FROM v_customer_360
GROUP BY tenure_group
ORDER BY
    CASE tenure_group
        WHEN '0-12 months' THEN 1
        WHEN '13-24 months' THEN 2
        WHEN '25-48 months' THEN 3
        ELSE 4
    END;

-- =====================================================
-- Ödeme Yöntemine Göre Churn Analizi
-- =====================================================
CREATE OR REPLACE VIEW v_churn_by_payment AS
SELECT
    payment_method,
    COUNT(*) as customer_count,
    SUM(CASE WHEN churned THEN 1 ELSE 0 END) as churned_count,
    ROUND(100.0 * SUM(CASE WHEN churned THEN 1 ELSE 0 END) / COUNT(*), 2) as churn_rate,
    ROUND(AVG(total_charges), 2) as avg_total_charges
FROM v_customer_360
GROUP BY payment_method
ORDER BY churn_rate DESC;

-- =====================================================
-- Yüksek Riskli Müşteriler
-- Churn etmemiş ama yüksek risk skoru olan müşteriler
-- =====================================================
CREATE OR REPLACE VIEW v_high_risk_customers AS
SELECT
    customer_id,
    gender,
    tenure_months,
    tenure_group,
    contract_type,
    monthly_charges,
    service_count,
    has_premium_support,
    contract_risk_score,
    payment_risk_score,
    ROUND((contract_risk_score + payment_risk_score) / 2, 2) as combined_risk_score
FROM v_customer_360
WHERE churned = false
  AND (contract_risk_score >= 0.8 OR payment_risk_score >= 0.8)
ORDER BY (contract_risk_score + payment_risk_score) DESC;

-- =====================================================
-- Son Tahminler Görünümü
-- =====================================================
CREATE OR REPLACE VIEW v_latest_predictions AS
SELECT DISTINCT ON (customer_id)
    p.customer_id,
    p.churn_probability,
    p.prediction,
    p.risk_level,
    p.risk_factors,
    p.model_version,
    p.predicted_at,
    c.tenure_months,
    c.gender,
    b.monthly_charges,
    b.contract_type
FROM predictions p
LEFT JOIN customers c ON p.customer_id = c.customer_id
LEFT JOIN billing b ON p.customer_id = b.customer_id
ORDER BY p.customer_id, p.predicted_at DESC;

-- =====================================================
-- Model Performans Özeti
-- =====================================================
CREATE OR REPLACE VIEW v_model_performance AS
SELECT
    model_name,
    model_version,
    algorithm,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    roc_auc,
    training_date,
    is_active,
    CASE
        WHEN is_active THEN 'Active'
        ELSE 'Inactive'
    END as status
FROM model_registry
ORDER BY training_date DESC;

-- =====================================================
-- Günlük Churn Trendi
-- =====================================================
CREATE OR REPLACE VIEW v_churn_trend AS
SELECT
    DATE(churn_date) as churn_day,
    COUNT(*) as churn_count,
    STRING_AGG(churn_reason, ', ') as reasons
FROM churn_labels
WHERE churned = true AND churn_date IS NOT NULL
GROUP BY DATE(churn_date)
ORDER BY churn_day DESC;

-- =====================================================
-- ML için Feature Matrix View
-- Model eğitimi için kullanılacak özellikler
-- =====================================================
CREATE OR REPLACE VIEW v_ml_features AS
SELECT
    customer_id,

    -- Encoded categorical features
    CASE gender WHEN 'Male' THEN 1 ELSE 0 END as gender_encoded,
    CASE WHEN senior_citizen THEN 1 ELSE 0 END as senior_citizen_encoded,
    CASE WHEN partner THEN 1 ELSE 0 END as partner_encoded,
    CASE WHEN dependents THEN 1 ELSE 0 END as dependents_encoded,

    -- Numerical features
    tenure_months,
    monthly_charges,
    total_charges,
    service_count,

    -- Derived features
    COALESCE(avg_monthly_charge, monthly_charges) as avg_monthly_charge,
    contract_risk_score,
    payment_risk_score,
    CASE WHEN has_premium_support THEN 1 ELSE 0 END as has_premium_support_encoded,

    -- Internet service encoding
    CASE internet_service
        WHEN 'Fiber optic' THEN 2
        WHEN 'DSL' THEN 1
        ELSE 0
    END as internet_service_encoded,

    -- Contract encoding
    CASE contract_type
        WHEN 'Two year' THEN 2
        WHEN 'One year' THEN 1
        ELSE 0
    END as contract_encoded,

    -- Payment method encoding
    CASE payment_method
        WHEN 'Credit card (automatic)' THEN 3
        WHEN 'Bank transfer (automatic)' THEN 2
        WHEN 'Mailed check' THEN 1
        ELSE 0
    END as payment_encoded,

    -- Target variable
    CASE WHEN churned THEN 1 ELSE 0 END as churn_label

FROM v_customer_360;
