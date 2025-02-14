import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_utility_customer_data(num_customers=1000, seed=42):
    """
    Generate comprehensive utility customer dataset

    Args:
        num_customers: Number of customers to generate
        seed: Random seed for reproducibility

    Returns:
        tuple: (customer_df, usage_df) containing customer and usage data
    """
    np.random.seed(seed)

    # Generate base customer data
    data = {
        # Basic Information
        'account_number': [f'ACC{str(i).zfill(6)}' for i in range(num_customers)],
        'credit_score': np.clip(np.random.normal(680, 50, num_customers), 300, 850),

        # Usage Pattern Metrics
        'demand_response': np.random.beta(7, 3, num_customers),  # Higher is better
        'load_factor': np.random.beta(5, 3, num_customers),  # Ratio of avg to peak demand
        'peak_compliance': np.random.beta(6, 2, num_customers),  # How well they manage peak usage
        'seasonal_stability': np.random.beta(4, 3, num_customers),  # Consistency across seasons

        # Payment Characteristics
        'payment_history_score': np.random.beta(8, 2, num_customers) * 100,
        'budget_billing': np.random.choice([True, False], num_customers, p=[0.3, 0.7]),
        'payment_plan': np.random.choice([True, False], num_customers, p=[0.15, 0.85]),
        'channel_consistency': np.random.beta(8, 2, num_customers),
        'return_frequency': np.random.beta(2, 8, num_customers),  # Lower is better
        'autopay_status': np.random.choice(['enrolled', 'not_enrolled'], num_customers, p=[0.4, 0.6]),
        'late_payments_30': np.random.poisson(0.5, num_customers),
        'late_payments_60': np.random.poisson(0.2, num_customers),
        'late_payments_90': np.random.poisson(0.1, num_customers),

        # Service Quality Indicators
        'meter_access': np.random.beta(9, 1, num_customers),  # Higher is better
        'safety_compliance': np.random.beta(9, 1, num_customers),
        'service_calls': np.random.poisson(2, num_customers),
        'emergency_incidents': np.random.poisson(0.2, num_customers),
        'meter_reading_consistency': np.random.beta(8, 2, num_customers),
        'infrastructure_participation': np.random.choice([True, False], num_customers, p=[0.25, 0.75]),

        # Account Management
        'multi_service': np.random.choice([True, False], num_customers, p=[0.35, 0.65]),
        'years_of_service': np.random.gamma(3, 2, num_customers),
        'transfer_history': np.random.poisson(0.5, num_customers),
        'premise_correlation': np.random.beta(7, 3, num_customers),
        'authorized_users': np.random.poisson(1, num_customers) + 1,
        'paperless_billing': np.random.choice([True, False], num_customers, p=[0.6, 0.4]),

        # Geographic/Regional Factors
        'zip_code': np.random.choice([f"{i:05d}" for i in range(10000, 10100)], num_customers),
        'weather_sensitivity': np.random.beta(4, 4, num_customers),
        'regional_delinquency_rate': np.random.beta(2, 8, num_customers),

        # Financial Metrics
        'income': np.clip(np.random.normal(60000, 20000, num_customers), 20000, 200000),
        'debt_to_income': np.clip(np.random.beta(3, 7, num_customers), 0, 1),
        'utilization_rate': np.clip(np.random.beta(3, 5, num_customers), 0, 1),

        # Time-based Metrics
        'days_since_last_late': np.random.gamma(5, 30, num_customers),
        'average_monthly_bill': np.random.gamma(5, 100, num_customers),
        'peak_season_performance': np.random.choice(
            ['excellent', 'good', 'fair', 'poor'],
            num_customers,
            p=[0.3, 0.4, 0.2, 0.1]
        ),

        # Usage Metrics
        'consumption_volatility': np.random.beta(3, 6, num_customers),
        'peak_time_usage': np.random.beta(4, 4, num_customers),
        'off_peak_usage': np.random.beta(4, 4, num_customers),
        'usage_predictability': np.random.choice(
            ['highly_predictable', 'predictable', 'variable', 'highly_variable'],
            num_customers,
            p=[0.2, 0.4, 0.3, 0.1]
        )
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add calculated fields
    df['recent_payments_ontime'] = 1 - (df['late_payments_30'] / 12)  # Based on last 12 months
    df['risk_category'] = pd.qcut(df['credit_score'], q=5,
                                  labels=['Very High', 'High', 'Medium', 'Low', 'Very Low'])

    # Generate monthly usage history (last 12 months)
    usage_history = []
    for account in df['account_number']:
        base_usage = np.random.gamma(10, 100)  # Base usage level
        seasonal_factor = np.random.beta(3, 3)  # How much seasonality affects usage

        for month in range(12):
            # Create seasonal pattern with some randomness
            seasonal_usage = base_usage * (1 +
                                           seasonal_factor * np.sin(2 * np.pi * (month + 6) / 12))
            usage_history.append({
                'account_number': account,
                'month': month + 1,
                'usage': np.random.normal(seasonal_usage, seasonal_usage * 0.1)
            })

    # Create usage history DataFrame
    usage_df = pd.DataFrame(usage_history)

    return df, usage_df


def generate_intervention_data(customer_df: pd.DataFrame, seed=42) -> pd.DataFrame:
    """
    Generate intervention data for existing customers

    Args:
        customer_df: DataFrame containing customer data
        seed: Random seed for reproducibility

    Returns:
        DataFrame containing intervention data
    """
    np.random.seed(seed)

    intervention_types = [
        'payment_arrangement',
        'deposit_adjustment',
        'budget_billing',
        'energy_assistance',
        'payment_reminder',
        'disconnect_notice'
    ]

    intervention_records = []

    for _, customer in customer_df.iterrows():
        # Generate interventions based on risk score
        num_interventions = np.random.poisson(max(1, customer.get('risk_score', 50) / 20))

        for seq in range(num_interventions):
            intervention_date = datetime.now() - timedelta(days=np.random.randint(1, 365))

            intervention_records.append({
                'customer_id': customer['account_number'],
                'intervention_date': intervention_date.strftime('%Y-%m-%d'),
                'intervention_type': np.random.choice(intervention_types),
                'intervention_sequence': seq + 1,
                'success': np.random.random() > (customer.get('risk_score', 50) / 100),
                'days_to_resolution': np.random.randint(1, 60),
                'cost': np.random.uniform(10, 100),
                'zip_code': customer['zip_code']
            })

    return pd.DataFrame(intervention_records)


if __name__ == "__main__":
    # Generate data
    customer_df, usage_df = generate_utility_customer_data()

    # Generate intervention data
    intervention_df = generate_intervention_data(customer_df)

    # Save to CSV files
    customer_df.to_csv('utility_customer_data.csv', index=False)
    usage_df.to_csv('utility_usage_history.csv', index=False)
    intervention_df.to_csv('utility_intervention_data.csv', index=False)

    print(f"Generated data for {len(customer_df)} customers:")
    print("\nCustomer Data Sample:")
    print(customer_df.head())
    print("\nUsage History Sample:")
    print(usage_df.head())
    print("\nIntervention Data Sample:")
    print(intervention_df.head())

    # Print data statistics
    print("\nKey Statistics:")
    for column in ['credit_score', 'peak_compliance', 'payment_history_score',
                   'meter_access', 'years_of_service']:
        print(f"\n{column}:")
        print(customer_df[column].describe())