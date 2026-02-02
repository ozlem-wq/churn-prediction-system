-- Churn Prediction System - Seed Data
-- Sample data based on Telco Customer Churn dataset patterns

-- =====================================================
-- Insert Sample Customers
-- =====================================================
INSERT INTO customers (customer_id, gender, senior_citizen, partner, dependents, tenure_months) VALUES
('7590-VHVEG', 'Female', false, true, false, 1),
('5575-GNVDE', 'Male', false, false, false, 34),
('3668-QPYBK', 'Male', false, false, false, 2),
('7795-CFOCW', 'Male', false, false, false, 45),
('9237-HQITU', 'Female', false, false, false, 2),
('9305-CDSKC', 'Female', false, false, false, 8),
('1452-KIOVK', 'Male', false, false, true, 22),
('6713-OKOMC', 'Female', false, false, false, 10),
('7892-POOKP', 'Female', true, true, false, 28),
('6388-TABGU', 'Male', false, true, true, 62),
('9763-GRSKD', 'Male', true, true, false, 13),
('7469-LKBCI', 'Male', false, false, false, 16),
('8091-TTVAX', 'Male', false, true, false, 58),
('0280-XJGEX', 'Male', false, false, false, 49),
('5129-JLPIS', 'Male', false, false, false, 25),
('3655-SNQYZ', 'Female', true, true, true, 69),
('8191-XWSZG', 'Female', false, false, false, 52),
('9959-WOFKT', 'Male', false, false, true, 71),
('4190-MFLUW', 'Female', false, true, true, 10),
('4183-MYFRB', 'Female', false, false, false, 21)
ON CONFLICT (customer_id) DO NOTHING;

-- =====================================================
-- Insert Services Data
-- =====================================================
INSERT INTO services (customer_id, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies) VALUES
('7590-VHVEG', false, 'No phone service', 'DSL', 'No', 'Yes', 'No', 'No', 'No', 'No'),
('5575-GNVDE', true, 'No', 'DSL', 'Yes', 'No', 'Yes', 'No', 'No', 'No'),
('3668-QPYBK', true, 'No', 'DSL', 'Yes', 'Yes', 'No', 'No', 'No', 'No'),
('7795-CFOCW', false, 'No phone service', 'DSL', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No'),
('9237-HQITU', true, 'No', 'Fiber optic', 'No', 'No', 'No', 'No', 'No', 'No'),
('9305-CDSKC', true, 'Yes', 'Fiber optic', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes'),
('1452-KIOVK', true, 'Yes', 'Fiber optic', 'No', 'Yes', 'No', 'No', 'Yes', 'No'),
('6713-OKOMC', false, 'No phone service', 'DSL', 'Yes', 'No', 'No', 'No', 'No', 'No'),
('7892-POOKP', true, 'Yes', 'Fiber optic', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes'),
('6388-TABGU', true, 'No', 'DSL', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No'),
('9763-GRSKD', true, 'No', 'DSL', 'Yes', 'No', 'No', 'No', 'No', 'No'),
('7469-LKBCI', true, 'No', 'No', 'No internet service', 'No internet service', 'No internet service', 'No internet service', 'No internet service', 'No internet service'),
('8091-TTVAX', true, 'Yes', 'Fiber optic', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes'),
('0280-XJGEX', true, 'Yes', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'),
('5129-JLPIS', true, 'No', 'Fiber optic', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes'),
('3655-SNQYZ', true, 'Yes', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'),
('8191-XWSZG', true, 'No', 'No', 'No internet service', 'No internet service', 'No internet service', 'No internet service', 'No internet service', 'No internet service'),
('9959-WOFKT', true, 'Yes', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes'),
('4190-MFLUW', true, 'No', 'DSL', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No'),
('4183-MYFRB', true, 'No', 'Fiber optic', 'No', 'No', 'No', 'No', 'No', 'No')
ON CONFLICT (customer_id) DO NOTHING;

-- =====================================================
-- Insert Billing Data
-- =====================================================
INSERT INTO billing (customer_id, contract_type, paperless_billing, payment_method, monthly_charges, total_charges) VALUES
('7590-VHVEG', 'Month-to-month', true, 'Electronic check', 29.85, 29.85),
('5575-GNVDE', 'One year', false, 'Mailed check', 56.95, 1889.50),
('3668-QPYBK', 'Month-to-month', true, 'Mailed check', 53.85, 108.15),
('7795-CFOCW', 'One year', false, 'Bank transfer (automatic)', 42.30, 1840.75),
('9237-HQITU', 'Month-to-month', true, 'Electronic check', 70.70, 151.65),
('9305-CDSKC', 'Month-to-month', true, 'Electronic check', 99.65, 820.50),
('1452-KIOVK', 'Month-to-month', true, 'Credit card (automatic)', 89.10, 1949.40),
('6713-OKOMC', 'Month-to-month', false, 'Mailed check', 29.75, 301.90),
('7892-POOKP', 'Month-to-month', true, 'Electronic check', 104.80, 3046.05),
('6388-TABGU', 'One year', true, 'Bank transfer (automatic)', 56.15, 3487.95),
('9763-GRSKD', 'Month-to-month', true, 'Electronic check', 49.95, 587.45),
('7469-LKBCI', 'Two year', false, 'Credit card (automatic)', 18.95, 326.80),
('8091-TTVAX', 'One year', false, 'Credit card (automatic)', 100.35, 5681.10),
('0280-XJGEX', 'One year', true, 'Bank transfer (automatic)', 103.70, 5036.30),
('5129-JLPIS', 'Month-to-month', true, 'Electronic check', 105.50, 2686.05),
('3655-SNQYZ', 'Two year', true, 'Credit card (automatic)', 113.25, 7895.15),
('8191-XWSZG', 'One year', false, 'Mailed check', 20.65, 1022.95),
('9959-WOFKT', 'Two year', true, 'Bank transfer (automatic)', 106.70, 7382.25),
('4190-MFLUW', 'Month-to-month', false, 'Credit card (automatic)', 62.25, 638.10),
('4183-MYFRB', 'Month-to-month', true, 'Electronic check', 69.40, 1497.40)
ON CONFLICT (customer_id) DO NOTHING;

-- =====================================================
-- Insert Churn Labels
-- =====================================================
INSERT INTO churn_labels (customer_id, churned, churn_date, churn_reason) VALUES
('7590-VHVEG', false, NULL, NULL),
('5575-GNVDE', false, NULL, NULL),
('3668-QPYBK', true, '2024-03-15', 'Competitor made better offer'),
('7795-CFOCW', false, NULL, NULL),
('9237-HQITU', true, '2024-02-20', 'Poor customer support'),
('9305-CDSKC', true, '2024-04-01', 'Price too high'),
('1452-KIOVK', false, NULL, NULL),
('6713-OKOMC', false, NULL, NULL),
('7892-POOKP', true, '2024-01-10', 'Moved to different area'),
('6388-TABGU', false, NULL, NULL),
('9763-GRSKD', true, '2024-03-25', 'Poor network quality'),
('7469-LKBCI', false, NULL, NULL),
('8091-TTVAX', false, NULL, NULL),
('0280-XJGEX', false, NULL, NULL),
('5129-JLPIS', true, '2024-02-28', 'Competitor made better offer'),
('3655-SNQYZ', false, NULL, NULL),
('8191-XWSZG', false, NULL, NULL),
('9959-WOFKT', false, NULL, NULL),
('4190-MFLUW', true, '2024-04-10', 'Service not needed anymore'),
('4183-MYFRB', true, '2024-03-05', 'Poor customer support')
ON CONFLICT (customer_id) DO NOTHING;

-- =====================================================
-- Insert Initial Model Registry Entry
-- =====================================================
INSERT INTO model_registry (model_name, model_version, model_path, algorithm, is_active) VALUES
('churn_classifier', 'v0.0.1', 'models/placeholder.pkl', 'placeholder', false)
ON CONFLICT (model_name, model_version) DO NOTHING;
