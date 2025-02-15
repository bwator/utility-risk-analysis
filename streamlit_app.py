import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import existing project modules
from utility_data_generator import generate_utility_customer_data, generate_intervention_data
from comprehensive_utility_risk_calculator import ComprehensiveUtilityRiskCalculator
from risk_analysis_framework import analyze_risk_data
from main import merge_customer_and_risk_data

# Streamlit page configuration
st.set_page_config(page_title="Utility Risk Analysis", layout="wide")


def generate_data():
    """Generate comprehensive utility customer dataset"""
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Generate data
    st.subheader("1. Data Generation")
    num_customers = st.slider("Number of Customers", 100, 5000, 1000)

    with st.spinner('Generating Customer Data...'):
        customer_df, usage_df = generate_utility_customer_data(num_customers=num_customers)
        intervention_df = generate_intervention_data(customer_df)

        # Save data
        customer_path = os.path.join(output_dir, "utility_customer_data.csv")
        usage_path = os.path.join(output_dir, "utility_usage_data.csv")
        intervention_path = os.path.join(output_dir, "intervention_data.csv")

        customer_df.to_csv(customer_path, index=False)
        usage_df.to_csv(usage_path, index=False)
        intervention_df.to_csv(intervention_path, index=False)

    st.success("Data Generation Complete!")
    return customer_df, usage_df, intervention_df


def calculate_risk_scores(customer_df):
    """Calculate risk scores for customers"""
    st.subheader("2. Risk Score Calculation")

    with st.spinner('Calculating Risk Scores...'):
        # Use existing risk calculator
        calculator = ComprehensiveUtilityRiskCalculator()

        # Temporary paths for calculation
        output_dir = "output"
        customer_path = os.path.join(output_dir, "utility_customer_data.csv")
        risk_scores_path = os.path.join(output_dir, "risk_scores.csv")
        merged_data_path = os.path.join(output_dir, "customer_data_risk.csv")

        # Process customer file
        calculator.process_customer_file(customer_path, risk_scores_path)

        # Merge data
        merge_customer_and_risk_data(customer_path, risk_scores_path, merged_data_path)

        # Read merged data
        merged_df = pd.read_csv(merged_data_path)

    st.success("Risk Scores Calculated!")
    return merged_df


def create_risk_dashboard(merged_df):
    """Create interactive Streamlit dashboard for risk analysis"""
    st.title("Utility Customer Risk Analysis Dashboard")

    # Sidebar for navigation
    analysis_type = st.sidebar.radio("Choose Analysis", [
        "Overview",
        "Risk Distribution",
        "Credit Analysis",
        "Multi-Factor Analysis",  # New view
        "Temporal Analysis",  # New view
        "Geographic Insights",
        "Delinquency Analysis",  # New view
        "Predictive Analysis",  # New view
        "Intervention Strategy"
    ])

    # Add implementations for new views similar to existing ones
    if analysis_type == "Multi-Factor Analysis":
        st.header("Payment and Usage Pattern Analysis")
        fig = px.scatter(merged_df,
                         x='payment_history_score',
                         y='peak_compliance',
                         color='risk_category',
                         size='average_monthly_bill',
                         title='Payment History vs Peak Usage Compliance')
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Temporal Analysis":
        st.header("Customer Tenure and Payment Behavior")
        # Calculate customer tenure groups
        merged_df['tenure_group'] = pd.qcut(
            merged_df['years_of_service'],
            q=5,
            labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        )

        # Payment History by Tenure
        fig = px.box(merged_df,
                     x='tenure_group',
                     y='payment_history_score',
                     title='Payment History Scores by Customer Tenure')
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Delinquency Analysis":
        st.header("Delinquency Risk Patterns")

        # Late Payment Analysis
        late_payment_data = merged_df[['late_payments_30', 'late_payments_60', 'late_payments_90']]
        late_payment_melted = late_payment_data.melt(var_name='Late Payment Category', value_name='Count')

        fig = px.box(late_payment_melted,
                     x='Late Payment Category',
                     y='Count',
                     title='Distribution of Late Payments')
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Predictive Analysis":
        st.header("Delinquency Prediction Insights")

        # Credit Score vs Delinquency Probability
        fig = px.scatter(merged_df,
                         x='credit_score',
                         y='risk_score',
                         color='risk_category',
                         title='Credit Score and Predicted Risk')
        st.plotly_chart(fig, use_container_width=True)

        # Highlight high-risk customers
        st.subheader("High-Risk Customers")
        high_risk = merged_df[merged_df['risk_score'] > merged_df['risk_score'].quantile(0.9)]
        st.dataframe(high_risk[['account_number', 'credit_score', 'risk_score', 'payment_history_score']])

    # Existing views remain the same...


def main():
    try:
        import os

        st.sidebar.header("Risk Analysis Workflow")

        st.title("Utility Customer Risk Analysis Dashboard")
        st.write("Attempting to generate or load dataset...")

        # Create output directory
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Always generate new dataset
        try:
            # Generate customer data
            customer_df, usage_df = generate_utility_customer_data(num_customers=1000)
            st.write("Customer data generated")

            # Generate intervention data
            intervention_df = generate_intervention_data(customer_df)
            st.write("Intervention data generated")

            # Save generated data
            customer_path = os.path.join(output_dir, "utility_customer_data.csv")
            usage_path = os.path.join(output_dir, "utility_usage_data.csv")
            intervention_path = os.path.join(output_dir, "intervention_data.csv")

            customer_df.to_csv(customer_path, index=False)
            usage_df.to_csv(usage_path, index=False)
            intervention_df.to_csv(intervention_path, index=False)

            # Calculate risk scores
            calculator = ComprehensiveUtilityRiskCalculator()
            risk_scores_path = os.path.join(output_dir, "risk_scores.csv")
            merged_data_path = os.path.join(output_dir, "customer_data_risk.csv")

            calculator.process_customer_file(customer_path, risk_scores_path)
            merge_customer_and_risk_data(customer_path, risk_scores_path, merged_data_path)

            # Load merged data
            merged_df = pd.read_csv(merged_data_path)
            st.write("Risk scores calculated")

        except Exception as gen_error:
            st.error(f"Error generating dataset: {gen_error}")
            return

        # Create dashboard with generated data
        create_risk_dashboard(merged_df)

    except Exception as e:
        st.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()