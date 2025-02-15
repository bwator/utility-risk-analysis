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
        "Geographic Insights",
        "Intervention Strategy"
    ])

    if analysis_type == "Overview":
        st.header("Customer Risk Portfolio")

        # Key Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", len(merged_df))
        with col2:
            st.metric("Average Risk Score", f"{merged_df['risk_score'].mean():.2f}")
        with col3:
            st.metric("High Risk Customers", len(merged_df[merged_df['risk_score'] > 70]))

        # Risk Score Distribution
        fig = px.histogram(merged_df, x='risk_score',
                           color_discrete_sequence=['blue'],
                           title='Distribution of Customer Risk Scores')
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Risk Distribution":
        st.header("Detailed Risk Stratification")

        # Credit Score vs Risk Score
        fig = px.scatter(merged_df,
                         x='credit_score',
                         y='risk_score',
                         color='risk_category',
                         title='Relationship between Credit Score and Risk Score')
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Credit Analysis":
        st.header("Credit Performance Insights")

        # Payment History Analysis
        fig = px.box(merged_df,
                     x='risk_category',
                     y='payment_history_score',
                     title='Payment History Scores Across Risk Categories')
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Geographic Insights":
        st.header("Geographic Risk Patterns")

        # Risk by ZIP Code
        risk_by_zip = merged_df.groupby('zip_code')['risk_score'].mean().reset_index()
        fig = px.bar(risk_by_zip.nlargest(20, 'risk_score'),
                     x='zip_code',
                     y='risk_score',
                     title='Top 20 ZIP Codes by Average Risk Score')
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Intervention Strategy":
        st.header("Targeted Intervention Recommendations")

        # High-Risk Customers Table
        high_risk = merged_df[merged_df['risk_score'] > 70].nlargest(10, 'risk_score')
        st.dataframe(high_risk[['account_number', 'credit_score', 'risk_score', 'payment_history_score']])


def main():
    # Data Generation Stage
    st.sidebar.header("Risk Analysis Workflow")
    if st.sidebar.button("Generate New Dataset"):
        customer_df, usage_df, intervention_df = generate_data()

    # Risk Score Calculation Stage
    if st.sidebar.button("Calculate Risk Scores"):
        merged_df = calculate_risk_scores(customer_df)
        create_risk_dashboard(merged_df)

    # Directly load existing data if available
    try:
        merged_df = pd.read_csv("output/customer_data_risk.csv")
        create_risk_dashboard(merged_df)
    except FileNotFoundError:
        st.warning("No existing risk data found. Please generate a dataset.")


# if __name__ == "__main__":
#    main()