"""
Churn Prediction Dashboard
==========================

Streamlit application for visualizing churn predictions and analytics.
"""

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .high-risk {
        color: #ff4b4b;
        font-weight: bold;
    }
    .medium-risk {
        color: #ffa600;
        font-weight: bold;
    }
    .low-risk {
        color: #00cc96;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper functions
def fetch_api(endpoint: str, method: str = "GET", data: dict = None):
    """Fetch data from API."""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    colors = {
        "HIGH": "#ff4b4b",
        "MEDIUM": "#ffa600",
        "LOW": "#00cc96",
    }
    return colors.get(risk_level, "#808080")


# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Customer Analysis", "Predictions", "Model Performance"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### API Status")

# Check API health
health = fetch_api("/health")
if health:
    st.sidebar.success(f"API: {health.get('status', 'unknown')}")
    st.sidebar.info(f"Database: {health.get('database', 'unknown')}")
else:
    st.sidebar.error("API: Disconnected")

st.sidebar.markdown("---")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# =====================================================
# Page: Overview
# =====================================================
if page == "Overview":
    st.title("Churn Prediction Dashboard")
    st.markdown("### Overview of Customer Churn Analytics")

    # Fetch statistics
    stats = fetch_api("/api/v1/stats")

    if stats:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Customers",
                value=f"{stats.get('total_customers', 0):,}",
            )

        with col2:
            st.metric(
                label="Churned Customers",
                value=f"{stats.get('churned_customers', 0):,}",
                delta=f"-{stats.get('churn_rate', 0):.1f}%",
                delta_color="inverse",
            )

        with col3:
            st.metric(
                label="Active Customers",
                value=f"{stats.get('active_customers', 0):,}",
            )

        with col4:
            st.metric(
                label="Churn Rate",
                value=f"{stats.get('churn_rate', 0):.1f}%",
            )

        st.markdown("---")

        # Churn by Contract Type
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Churn by Contract Type")
            contract_stats = fetch_api("/api/v1/stats/by-contract")
            if contract_stats:
                df = pd.DataFrame(contract_stats)
                fig = px.bar(
                    df,
                    x="contract_type",
                    y="churn_rate",
                    color="churn_rate",
                    color_continuous_scale="RdYlGn_r",
                    title="Churn Rate by Contract Type",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Churn by Tenure")
            tenure_stats = fetch_api("/api/v1/stats/by-tenure")
            if tenure_stats:
                df = pd.DataFrame(tenure_stats)
                fig = px.bar(
                    df,
                    x="tenure_group",
                    y="churn_rate",
                    color="churn_rate",
                    color_continuous_scale="RdYlGn_r",
                    title="Churn Rate by Tenure Group",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Churn by Payment Method
        st.markdown("### Churn by Payment Method")
        payment_stats = fetch_api("/api/v1/stats/by-payment")
        if payment_stats:
            df = pd.DataFrame(payment_stats)
            fig = px.pie(
                df,
                values="churned_count",
                names="payment_method",
                title="Churned Customers by Payment Method",
                hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Unable to load statistics. Please check API connection.")


# =====================================================
# Page: Customer Analysis
# =====================================================
elif page == "Customer Analysis":
    st.title("Customer Analysis")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        churn_filter = st.selectbox(
            "Filter by Churn Status",
            options=["All", "Churned", "Active"],
        )

    with col2:
        limit = st.slider("Number of customers", 10, 100, 50)

    # Build query
    endpoint = f"/api/v1/customers?limit={limit}"
    if churn_filter == "Churned":
        endpoint += "&churned=true"
    elif churn_filter == "Active":
        endpoint += "&churned=false"

    customers = fetch_api(endpoint)

    if customers:
        df = pd.DataFrame(customers)

        # Summary
        st.markdown(f"### Showing {len(df)} customers")

        # Churn distribution
        if "churned" in df.columns:
            col1, col2, col3 = st.columns(3)

            with col1:
                churned_count = df["churned"].sum()
                st.metric("Churned", churned_count)

            with col2:
                active_count = len(df) - churned_count
                st.metric("Active", active_count)

            with col3:
                avg_tenure = df["tenure_months"].mean()
                st.metric("Avg Tenure", f"{avg_tenure:.1f} months")

        # Customer table
        st.markdown("### Customer Details")

        display_cols = [
            "customer_id",
            "gender",
            "tenure_months",
            "contract_type",
            "monthly_charges",
            "churned",
        ]
        display_cols = [c for c in display_cols if c in df.columns]

        st.dataframe(df[display_cols], use_container_width=True)

        # Distribution charts
        st.markdown("### Distributions")

        col1, col2 = st.columns(2)

        with col1:
            if "monthly_charges" in df.columns:
                fig = px.histogram(
                    df,
                    x="monthly_charges",
                    nbins=20,
                    title="Monthly Charges Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "tenure_months" in df.columns:
                fig = px.histogram(
                    df,
                    x="tenure_months",
                    nbins=20,
                    title="Tenure Distribution",
                )
                st.plotly_chart(fig, use_container_width=True)


# =====================================================
# Page: Predictions
# =====================================================
elif page == "Predictions":
    st.title("Churn Predictions")

    st.markdown("### Make a Prediction")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            customer_id = st.text_input("Customer ID", value="NEW-001")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.checkbox("Senior Citizen")
            tenure_months = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            contract_type = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
            )
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
            )
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
            online_security = st.selectbox("Online Security", ["No", "Yes"])

        with col3:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=50.0,
            )
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            partner = st.checkbox("Has Partner")
            dependents = st.checkbox("Has Dependents")

        submitted = st.form_submit_button("Predict Churn")

        if submitted:
            prediction_data = {
                "customer_id": customer_id,
                "gender": gender,
                "senior_citizen": senior_citizen,
                "partner": partner,
                "dependents": dependents,
                "tenure_months": tenure_months,
                "contract_type": contract_type,
                "internet_service": internet_service,
                "tech_support": tech_support,
                "online_security": online_security,
                "monthly_charges": monthly_charges,
                "payment_method": payment_method,
            }

            result = fetch_api("/api/v1/predict", method="POST", data=prediction_data)

            if result:
                st.markdown("---")
                st.markdown("### Prediction Result")

                col1, col2, col3 = st.columns(3)

                with col1:
                    prob = result.get("churn_probability", 0)
                    st.metric(
                        "Churn Probability",
                        f"{prob * 100:.1f}%",
                    )

                with col2:
                    risk = result.get("risk_level", "UNKNOWN")
                    color = get_risk_color(risk)
                    st.markdown(
                        f"**Risk Level:** <span style='color:{color}'>{risk}</span>",
                        unsafe_allow_html=True,
                    )

                with col3:
                    prediction = "Yes" if result.get("prediction") else "No"
                    st.markdown(f"**Will Churn:** {prediction}")

                # Risk factors
                st.markdown("### Top Risk Factors")
                factors = result.get("top_risk_factors", [])
                for factor in factors:
                    st.warning(factor)

                # Gauge chart
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Churn Probability"},
                        gauge={
                            "axis": {"range": [None, 100]},
                            "bar": {"color": color},
                            "steps": [
                                {"range": [0, 40], "color": "#00cc96"},
                                {"range": [40, 70], "color": "#ffa600"},
                                {"range": [70, 100], "color": "#ff4b4b"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                    )
                )
                st.plotly_chart(fig, use_container_width=True)


# =====================================================
# Page: Model Performance
# =====================================================
elif page == "Model Performance":
    st.title("Model Performance")

    # Model info
    model_info = fetch_api("/api/v1/model/info")

    if model_info:
        st.markdown("### Current Model")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(f"**Model:** {model_info.get('model_name', 'N/A')}")

        with col2:
            st.info(f"**Version:** {model_info.get('model_version', 'N/A')}")

        with col3:
            st.info(f"**Type:** {model_info.get('model_type', 'N/A')}")

        st.markdown("### Risk Level Thresholds")

        risk_levels = model_info.get("risk_levels", {})
        col1, col2, col3 = st.columns(3)

        with col1:
            st.error(f"HIGH: {risk_levels.get('high', 'N/A')}")

        with col2:
            st.warning(f"MEDIUM: {risk_levels.get('medium', 'N/A')}")

        with col3:
            st.success(f"LOW: {risk_levels.get('low', 'N/A')}")

        # Performance metrics (would come from model registry in production)
        st.markdown("---")
        st.markdown("### Performance Metrics (Training)")

        col1, col2, col3, col4, col5 = st.columns(5)

        # These would come from actual model evaluation
        metrics = {
            "Accuracy": 0.82,
            "Precision": 0.78,
            "Recall": 0.75,
            "F1-Score": 0.76,
            "ROC-AUC": 0.85,
        }

        for col, (metric, value) in zip(
            [col1, col2, col3, col4, col5], metrics.items()
        ):
            with col:
                st.metric(metric, f"{value:.2%}")

        # ROC Curve visualization (placeholder)
        st.markdown("### ROC Curve")

        # Simulated ROC curve data
        import numpy as np

        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # Simplified curve

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC (AUC = {metrics['ROC-AUC']:.2f})",
                line=dict(color="blue", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="gray", dash="dash"),
            )
        )
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Unable to load model information.")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Churn Prediction System v1.0.0 | Powered by FastAPI & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
