import os
import pandas as pd
from datetime import datetime
from utility_data_generator import generate_utility_customer_data, generate_intervention_data
from comprehensive_utility_risk_calculator import ComprehensiveUtilityRiskCalculator
from risk_analysis_framework import analyze_risk_data
from risk_dashboard import EnhancedRiskDashboard  # Import the class instead of the function


def merge_customer_and_risk_data(customer_data_path: str, risk_scores_path: str, output_path: str) -> None:
    """
    Merge customer data with calculated risk scores

    Args:
        customer_data_path: Path to customer data CSV
        risk_scores_path: Path to risk scores CSV
        output_path: Path to save merged data
    """
    try:
        # Load the data
        customer_df = pd.read_csv(customer_data_path)
        risk_df = pd.read_csv(risk_scores_path)

        # Merge on customer_id/account_number
        merged_df = customer_df.merge(
            risk_df,
            left_on='account_number',
            right_on='customer_id',
            how='left'
        )

        # Drop duplicate customer_id column if it exists
        if 'customer_id' in merged_df.columns:
            merged_df.drop('customer_id', axis=1, inplace=True)

        # Save merged data
        merged_df.to_csv(output_path, index=False)

        # Print summary statistics of merged data
        print("\nMerged Data Summary:")
        print(f"Total customers: {len(merged_df)}")
        print("\nRisk Score Distribution:")
        print(merged_df['risk_score'].describe())

    except Exception as e:
        print(f"Error merging data: {str(e)}")
        raise


def main():
    """Main execution flow"""
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Add timestamp to filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        print("\n=== Starting Utility Risk Analysis Process ===")

        # Step 1: Generate Customer Data
        print("\n1. Generating Customer Data...")
        customer_df, usage_df = generate_utility_customer_data(num_customers=1000)

        # Save generated data
        customer_data_path = os.path.join(output_dir, f"utility_customer_data_{timestamp}.csv")
        usage_data_path = os.path.join(output_dir, f"utility_usage_data_{timestamp}.csv")

        customer_df.to_csv(customer_data_path, index=False)
        usage_df.to_csv(usage_data_path, index=False)

        print(f"Generated data saved to:")
        print(f"- Customer data: {customer_data_path}")
        print(f"- Usage data: {usage_data_path}")

        # Step 2: Generate Intervention Data
        print("\n2. Generating Intervention Data...")
        intervention_df = generate_intervention_data(customer_df)
        intervention_data_path = os.path.join(output_dir, f"intervention_data_{timestamp}.csv")
        intervention_df.to_csv(intervention_data_path, index=False)
        print(f"- Intervention data: {intervention_data_path}")

        # Step 3: Calculate Risk Scores
        print("\n3. Calculating Risk Scores...")
        calculator = ComprehensiveUtilityRiskCalculator()
        risk_scores_path = os.path.join(output_dir, f"risk_scores_{timestamp}.csv")
        calculator.process_customer_file(customer_data_path, risk_scores_path)

        # Step 4: Merge Data
        print("\n4. Merging Customer Data with Risk Scores...")
        merged_data_path = os.path.join(output_dir, f"customer_data_risk_{timestamp}.csv")
        merge_customer_and_risk_data(customer_data_path, risk_scores_path, merged_data_path)

        # Step 5: Perform Risk Analysis
        print("\n5. Performing Comprehensive Risk Analysis...")
        analysis_output_dir = os.path.join(output_dir, f"analysis_results_{timestamp}")
        analysis_results = analyze_risk_data(merged_data_path, analysis_output_dir)

        # Step 6: Launch Interactive Dashboard
        print("\n6. Launching Interactive Dashboard...")
        try:
            dashboard = EnhancedRiskDashboard(merged_data_path)
            print(f"\nDashboard initialized. Access it at http://localhost:8050")
            print("Press Ctrl+C to stop the dashboard server.")
            dashboard.run_server(debug=False, port=8050)
        except Exception as e:
            print(f"Error launching dashboard: {e}")
            print("Analysis results are still available in the output directory")

        print("\n=== Process Complete ===")
        print("Files generated:")
        print(f"1. {customer_data_path}")
        print(f"2. {usage_data_path}")
        print(f"3. {intervention_data_path}")
        print(f"4. {risk_scores_path}")
        print(f"5. {merged_data_path}")
        print(f"6. {analysis_output_dir}/")

    except Exception as e:
        print(f"\nError in execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()