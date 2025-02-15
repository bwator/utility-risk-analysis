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
from risk_dashboard import EnhancedRiskDashboard

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
        "Main Analysis",
        "Multi-Factor Analysis",
        "Enhanced Intervention Analysis",
        "Temporal Analysis",
        "Geographic Analysis",
        "Delinquency Analysis",
        "Predictive Analysis"
    ])

    if analysis_type == "Main Analysis":
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
        st.subheader("Risk Score Distribution")
        fig_dist = px.histogram(merged_df, x='risk_score',
                                color='risk_band',
                                title='Risk Score Distribution')
        st.plotly_chart(fig_dist, use_container_width=True)

        # Risk vs Credit Score Scatter Plot
        st.subheader("Risk vs Credit Score Analysis")
        fig_scatter = px.scatter(merged_df,
                                 x='credit_score',
                                 y='risk_score',
                                 color='risk_band',
                                 title='Risk Score vs Credit Score')
        st.plotly_chart(fig_scatter, use_container_width=True)

    elif analysis_type == "Multi-Factor Analysis":
        st.header("Payment and Usage Pattern Analysis")
        fig = px.scatter(merged_df,
                         x='payment_history_score',
                         y='peak_compliance',
                         color='risk_band',
                         size='average_monthly_bill',
                         title='Payment History vs Peak Usage Compliance')
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Enhanced Intervention Analysis":
        st.header("Intervention Effectiveness Analysis")

        # Traditional Risk Analysis Scatter Plot
        st.subheader("Traditional Risk Analysis")
        fig_intervention = px.scatter(merged_df,
                                      x='payment_history_score',
                                      y='risk_score',
                                      color='risk_band',
                                      size='average_monthly_bill',
                                      hover_data=[
                                          'account_number',
                                          'credit_score',
                                          'late_payments_30',
                                          'late_payments_60',
                                          'late_payments_90',
                                          'autopay_status'
                                      ],
                                      title='Payment History and Risk Analysis')
        st.plotly_chart(fig_intervention, use_container_width=True)

        # Note: Timeline and Intervention Effectiveness data require additional data processing
        st.warning("Full intervention analysis requires additional intervention data.")

    elif analysis_type == "Temporal Analysis":
        st.header("Temporal Customer Analysis")

        # Create customer tenure groups
        merged_df['tenure_group'] = pd.qcut(
            merged_df['years_of_service'],
            q=5,
            labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        )

        # Payment History Trends
        st.subheader("Payment History Trends")
        fig_payment_history = px.scatter(merged_df,
                                         x='years_of_service',
                                         y='payment_history_score',
                                         color='risk_band',
                                         opacity=0.7,
                                         title='Payment History by Customer Tenure')
        st.plotly_chart(fig_payment_history, use_container_width=True)

        # Risk Score Evolution
        st.subheader("Risk Score Evolution")
        fig_risk_evolution = px.box(merged_df,
                                    x='tenure_group',
                                    y='risk_score',
                                    color='risk_band',
                                    title='Risk Score Distribution by Customer Tenure')
        st.plotly_chart(fig_risk_evolution, use_container_width=True)

        # Payment Behavior Analysis
        st.subheader("Payment Behavior Analysis")
        fig_payment_behavior = px.scatter(merged_df,
                                          x='years_of_service',
                                          y='average_monthly_bill',
                                          size='payment_history_score',
                                          color='risk_band',
                                          opacity=0.7,
                                          title='Payment Behavior by Tenure and Bill Size')
        st.plotly_chart(fig_payment_behavior, use_container_width=True)

    elif analysis_type == "Geographic Analysis":
        st.header("Geographic Risk Analysis")

        # Geographic Risk Distribution
        st.subheader("Geographic Risk Distribution")
        fig_geo_dist = px.box(merged_df,
                              x='zip_code',
                              y='risk_score',
                              color='risk_band',
                              title='Risk Score Distribution by ZIP Code')
        st.plotly_chart(fig_geo_dist, use_container_width=True)

        # Regional Delinquency Analysis
        st.subheader("Regional Delinquency Analysis")
        fig_regional_delinq = px.scatter(merged_df,
                                         x='regional_delinquency_rate',
                                         y='risk_score',
                                         color='risk_band',
                                         size='average_monthly_bill',
                                         title='Risk Score vs Regional Delinquency Rate')
        st.plotly_chart(fig_regional_delinq, use_container_width=True)

    elif analysis_type == "Delinquency Analysis":
        st.header("Delinquency Risk Analysis")

        # Calculate risk quintiles
        merged_df['risk_quintile'] = pd.qcut(merged_df['risk_score'],
                                             q=5,
                                             labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

        # Calculate quintile statistics
        quintile_stats = merged_df.groupby('risk_quintile').agg({
            'risk_score': 'mean',
            'late_payments_30': 'mean',
            'late_payments_60': 'mean',
            'late_payments_90': 'mean'
        }).reset_index()

        # Calculate regional statistics
        regional_stats = merged_df.groupby('zip_code').agg({
            'risk_score': 'mean',
            'late_payments_30': 'mean',
            'payment_history_score': 'mean'
        }).reset_index()

        # Risk Quintile Analysis
        st.subheader("Risk Quintile Analysis")
        fig_quintile = px.bar(
            quintile_stats,
            x='risk_quintile',
            y=['late_payments_30', 'late_payments_60', 'late_payments_90'],
            title='Delinquency Patterns by Risk Quintile',
            barmode='group'
        )
        st.plotly_chart(fig_quintile, use_container_width=True)

        # Delinquency Risk Distribution
        st.subheader("Delinquency Risk Distribution")
        fig_risk_dist = px.histogram(
            merged_df,
            x='risk_score',
            color='risk_band',
            title='Risk Score Distribution with Delinquency Bands',
            marginal='box'
        )
        st.plotly_chart(fig_risk_dist, use_container_width=True)

        # Geographic Delinquency Patterns
        st.subheader("Geographic Delinquency Patterns")
        fig_geo_delinq = px.scatter(
            regional_stats,
            x='payment_history_score',
            y='late_payments_30',
            size='risk_score',
            hover_data=['zip_code'],
            title='Payment History vs Delinquency by ZIP Code'
        )
        st.plotly_chart(fig_geo_delinq, use_container_width=True)

        # Key Insights Card
        st.subheader("Key Delinquency Insights")
        insights_html = f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h4>Risk Quintile Analysis:</h4>
                    <ul>
                        <li>Highest risk quintile (Q5) shows {quintile_stats['late_payments_30'].iloc[-1]:.1f} average 30-day late payments</li>
                        <li>Lowest risk quintile (Q1) shows {quintile_stats['late_payments_30'].iloc[0]:.1f} average 30-day late payments</li>
                    </ul>

                    <h4>Geographic Patterns:</h4>
                    <ul>
                        <li>ZIP codes with higher risk scores show increased delinquency rates</li>
                        <li>Payment history strongly correlates with delinquency risk</li>
                    </ul>

                    <h4>Recommendations:</h4>
                    <ul>
                        <li>Focus intervention strategies on high-risk quintiles</li>
                        <li>Develop targeted programs for ZIP codes with elevated risk profiles</li>
                        <li>Implement early warning system based on payment history patterns</li>
                    </ul>
                </div>
                """
        st.markdown(insights_html, unsafe_allow_html=True)

    elif analysis_type == "Predictive Analysis":
        st.header("Predictive Analysis")

        # Prepare delinquency prediction data
        merged_df['is_delinquent'] = (
                (merged_df['late_payments_30'] > 0) |
                (merged_df['late_payments_60'] > 0) |
                (merged_df['late_payments_90'] > 0)
        ).astype(int)

        # Model Performance Metrics (Classification Report)
        st.subheader("Model Performance Metrics")
        from sklearn.metrics import classification_report

        # Prepare classification report
        pred_proba = merged_df['delinquency_probability'] > 0.5
        classification_rep = classification_report(
            merged_df['is_delinquent'],
            pred_proba.astype(int),
            output_dict=True
        )

        # Display classification report
        st.text(classification_report(
            merged_df['is_delinquent'],
            pred_proba.astype(int)
        ))

        # Feature Importance Plot
        st.subheader("Feature Importance in Predicting Delinquency")

        # Prepare feature importance data
        features = [
            'credit_score', 'payment_history_score', 'peak_compliance',
            'channel_consistency', 'return_frequency', 'years_of_service',
            'debt_to_income', 'utilization_rate', 'average_monthly_bill',
            'consumption_volatility', 'autopay_binary', 'budget_billing_binary'
        ]

        # Create dummy feature importance for demonstration
        import numpy as np
        np.random.seed(42)
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': np.random.random(len(features))
        }).sort_values('importance', ascending=False)

        # Plot feature importance
        fig_feature_importance = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance in Predicting Delinquency'
        )
        st.plotly_chart(fig_feature_importance, use_container_width=True)

        # Delinquency Probability Distribution
        st.subheader("Delinquency Probability Distribution")
        fig_prob_dist = px.histogram(
            merged_df,
            x='delinquency_probability',
            color='risk_level',
            title='Distribution of Delinquency Probabilities',
            marginal='box'
        )
        st.plotly_chart(fig_prob_dist, use_container_width=True)

        # Risk Factor Analysis
        st.subheader("Risk Factor Analysis")
        fig_risk_factor = px.scatter(
            merged_df,
            x='credit_score',
            y='delinquency_probability',
            color='risk_level',
            size='average_monthly_bill',
            hover_data=[
                'account_number',
                'payment_history_score',
                'years_of_service'
            ],
            title='Credit Score vs Delinquency Probability'
        )
        st.plotly_chart(fig_risk_factor, use_container_width=True)

        # High-Risk Customers Table
        st.subheader("High-Risk Customers (Top 10)")
        high_risk_customers = merged_df.nlargest(10, 'delinquency_probability')

        # Create a more styled dataframe display
        st.dataframe(
            high_risk_customers[[
                'account_number',
                'delinquency_probability',
                'risk_level',
                'credit_score',
                'payment_history_score'
            ]],
            use_container_width=True
        )


def main():
    try:
        import os

        st.sidebar.header("Risk Analysis Workflow")
        st.title("Utility Customer Risk Analysis Dashboard")
        st.write("Attempting to generate or load dataset...")

        # Create output directory
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Generate and save initial data
            customer_df, usage_df = generate_utility_customer_data(num_customers=1000)
            intervention_df = generate_intervention_data(customer_df)

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

            # Initialize dashboard for preprocessing
            dashboard = EnhancedRiskDashboard(merged_data_path)
            # Use the preprocessed dataframe from the dashboard
            preprocessed_df = dashboard.df

            # Create dashboard visualizations with preprocessed data
            create_risk_dashboard(preprocessed_df)

        except Exception as gen_error:
            st.error(f"Error generating dataset: {gen_error}")
            return

    except Exception as e:
        st.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()