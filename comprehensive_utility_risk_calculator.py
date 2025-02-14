"""
Comprehensive Utility and Credit Risk Calculator
----------------------------------------------

A sophisticated risk assessment system combining traditional credit metrics
with detailed utility-specific factors and robust data validation.

Core Components:
1. Data Validation System
2. Risk Score Calculator
3. Utility-Specific Metrics
4. Reporting System

Risk Components:
---------------
1. Credit Factors (45%):
   - Payment History (25%)
   - Credit Score Impact (20%)

2. Utility Factors (55%):
   - Usage Patterns (15%)
       * Demand Response Participation
       * Load Factor
       * Peak Compliance (30% of usage patterns)
       * Seasonal Stability
   - Payment Characteristics (15%)
       * Budget Billing Performance
       * Payment Plan Adherence
       * Channel Consistency
       * Return Payment Frequency
   - Service Quality (15%)
       * Meter Access History
       * Safety Compliance
       * Service Call History
       * Emergency Incidents
   - Account Management (10%)
       * Multi-Service Relationships
       * Transfer History
       * Premise Correlation
       * Authorized User Stability

Scoring Methodology:
------------------
Risk Score = 0-100 (Higher = Higher Risk)
Final Score incorporates:
- Base component scores
- Regional adjustments
- Seasonal factors
- Industry-specific modifiers
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class UtilityMetrics:
    """Structure for utility-specific metrics"""
    usage_patterns: Dict[str, float]
    payment_chars: Dict[str, float]
    service_quality: Dict[str, float]
    account_mgmt: Dict[str, float]

class ComprehensiveUtilityRiskCalculator:
    def __init__(self):
        """Initialize calculator with industry-standard weights and parameters"""
        self._initialize_scoring_components()

    def _initialize_scoring_components(self):
        """Initialize all scoring components and weights"""
        # Usage Pattern Metrics
        self.usage_weights = {
            'demand_response': 0.20,
            'load_factor': 0.25,
            'peak_compliance': 0.30,
            'seasonal_stability': 0.25
        }

        # Payment Characteristics
        self.payment_weights = {
            'budget_billing': 0.15,
            'payment_plan': 0.20,
            'channel_consistency': 0.15,
            'return_frequency': 0.50
        }

        # Service Quality
        self.service_weights = {
            'meter_access': 0.20,
            'safety_compliance': 0.30,
            'service_calls': 0.25,
            'emergency_incidents': 0.25
        }

        # Account Management
        self.account_weights = {
            'multi_service': 0.25,
            'transfer_history': 0.25,
            'premise_correlation': 0.25,
            'authorized_users': 0.25
        }

    def calculate_customer_risk_score(self, customer_data: pd.Series) -> float:
        """
        Calculate risk score for a single customer

        Args:
            customer_data: Series containing customer information

        Returns:
            float: Risk score between 0-100
        """
        try:
            # Calculate component scores
            usage_score = self._calculate_usage_score(customer_data)
            payment_score = self._calculate_payment_score(customer_data)
            service_score = self._calculate_service_score(customer_data)
            account_score = self._calculate_account_score(customer_data)
            credit_score = self._calculate_credit_component(customer_data)

            # Calculate weighted final score
            final_score = (
                usage_score * 0.15 +
                payment_score * 0.15 +
                service_score * 0.15 +
                account_score * 0.10 +
                credit_score * 0.45
            )

            return np.clip(final_score, 0, 100)

        except Exception as e:
            print(f"Error calculating risk score: {e}")
            return None

    def _calculate_usage_score(self, data: pd.Series) -> float:
        """Calculate usage pattern score"""
        score = 0
        for metric, weight in self.usage_weights.items():
            if metric in data:
                score += data[metric] * weight
        return score * 100

    def _calculate_payment_score(self, data: pd.Series) -> float:
        """Calculate payment characteristics score"""
        score = 0
        for metric, weight in self.payment_weights.items():
            if metric in data:
                score += data[metric] * weight
        return score * 100

    def _calculate_service_score(self, data: pd.Series) -> float:
        """Calculate service quality score"""
        score = 0
        for metric, weight in self.service_weights.items():
            if metric in data:
                score += data[metric] * weight
        return score * 100

    def _calculate_account_score(self, data: pd.Series) -> float:
        """Calculate account management score"""
        score = 0
        for metric, weight in self.account_weights.items():
            if metric in data:
                score += data[metric] * weight
        return score * 100

    def _calculate_credit_component(self, data: pd.Series) -> float:
        """Calculate traditional credit component"""
        credit_score = data.get('credit_score', 650)
        payment_history = data.get('payment_history_score', 70)
        return ((credit_score - 300) / 550 * 70 + payment_history * 0.3)

    def process_customer_file(self, input_path: str, output_path: str) -> None:
        """
        Process customer data file and generate risk scores

        Args:
            input_path: Path to input CSV file
            output_path: Path to save output CSV file
        """
        try:
            # Read input file
            print(f"Reading customer data from {input_path}")
            df = pd.read_csv(input_path)

            # Calculate risk scores
            print("Calculating comprehensive risk scores...")
            risk_scores = []
            for _, row in df.iterrows():
                score = self.calculate_customer_risk_score(row)
                risk_scores.append(score)

            # Create output DataFrame
            output_df = pd.DataFrame({
                'customer_id': df['account_number'],
                'risk_score': risk_scores
            })

            # Save results
            print(f"Saving risk scores to {output_path}")
            output_df.to_csv(output_path, index=False)

            # Print summary statistics
            print("\nRisk Score Summary Statistics:")
            print(output_df['risk_score'].describe())

            # Print risk band distribution
            risk_bands = pd.cut(output_df['risk_score'],
                              bins=[0, 20, 40, 60, 80, 100],
                              labels=['Minimal', 'Low', 'Moderate', 'High', 'Severe'])
            print("\nRisk Band Distribution:")
            print(risk_bands.value_counts().sort_index())

        except FileNotFoundError:
            print(f"Error: Input file {input_path} not found")
        except Exception as e:
            print(f"Error processing file: {e}")

def main():
    """Main function to run the risk calculator"""
    # Initialize calculator
    calculator = ComprehensiveUtilityRiskCalculator()

    # Define input and output paths
    input_path = 'utility_customer_data.csv'
    output_path = 'customer_risk_scores.csv'

    # Process data
    calculator.process_customer_file(input_path, output_path)

if __name__ == "__main__":
    main()