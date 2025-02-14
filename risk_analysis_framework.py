import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from pathlib import Path


class RiskAnalysisFramework:
    def __init__(self, data_path: str):
        """
        Initialize analyzer with data from merged customer risk file

        Args:
            data_path: Path to merged customer risk data CSV
        """
        self.data = self._load_data(data_path)
        self._initialize_parameters()

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and prepare data for analysis"""
        try:
            df = pd.read_csv(data_path)
            print(f"Loaded data with {len(df)} records")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _initialize_parameters(self):
        """Initialize analysis parameters"""
        self.risk_bands = {
            'Minimal': (0, 20),
            'Low': (20, 40),
            'Moderate': (40, 60),
            'High': (60, 80),
            'Severe': (80, 100)
        }

        self.key_metrics = [
            'risk_score', 'credit_score', 'payment_history_score',
            'peak_compliance', 'utilization_rate', 'consumption_volatility'
        ]

    def analyze_risk_distribution(self, output_dir: str) -> Dict:
        """Analyze risk score distribution and create visualizations"""
        # Create risk bands
        self.data['risk_band'] = pd.cut(
            self.data['risk_score'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Minimal', 'Low', 'Moderate', 'High', 'Severe']
        )

        # Create distribution plot
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.data, x='risk_score', bins=30)
        plt.title('Risk Score Distribution')
        plt.savefig(f"{output_dir}/risk_distribution.png")
        plt.close()

        # Calculate distribution statistics
        stats = {
            'distribution': self.data['risk_score'].describe().to_dict(),
            'risk_bands': self.data['risk_band'].value_counts(dropna=False).to_dict()
        }

        return stats

    def analyze_key_correlations(self, output_dir: str) -> Dict:
        """Analyze correlations between risk factors"""
        # Calculate correlation matrix
        corr_matrix = self.data[self.key_metrics].corr()

        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Risk Factor Correlations')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_matrix.png")
        plt.close()

        # Find strongest correlations with risk score
        risk_correlations = corr_matrix['risk_score'].sort_values(ascending=False)

        return {
            'risk_correlations': risk_correlations.to_dict(),
            'correlation_matrix': corr_matrix.to_dict()
        }

    def analyze_risk_factors_by_band(self, output_dir: str) -> Dict:
        """Analyze key metrics across risk bands"""
        results = {}

        for metric in self.key_metrics[1:]:  # Exclude risk_score itself
            # Calculate stats by risk band
            band_stats = self.data.groupby('risk_band', observed=True)[metric].agg([
                'mean', 'std', 'min', 'max'
            ]).round(2)

            # Create box plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.data, x='risk_band', y=metric)
            plt.title(f'{metric} by Risk Band')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{metric}_by_band.png")
            plt.close()

            results[metric] = band_stats.to_dict()

        return results

    def analyze_usage_patterns(self, output_dir: str) -> Dict:
        """Analyze usage patterns and their relationship with risk"""
        usage_metrics = [
            'peak_compliance', 'consumption_volatility',
            'seasonal_stability', 'load_factor'
        ]

        results = {}

        for metric in usage_metrics:
            if metric in self.data.columns:
                # Calculate stats by risk band
                stats = self.data.groupby('risk_band', observed=True)[metric].mean()

                # Create visualization
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=self.data, x=metric, y='risk_score', alpha=0.5)
                plt.title(f'Risk Score vs {metric}')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/risk_vs_{metric}.png")
                plt.close()

                results[metric] = {
                    'band_averages': stats.to_dict(),
                    'correlation': self.data[metric].corr(self.data['risk_score'])
                }

        return results

    def generate_comprehensive_analysis(self, output_dir: str) -> Dict:
        """Generate comprehensive analysis report"""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Perform analyses
        distribution_analysis = self.analyze_risk_distribution(output_dir)
        correlation_analysis = self.analyze_key_correlations(output_dir)
        band_analysis = self.analyze_risk_factors_by_band(output_dir)
        usage_analysis = self.analyze_usage_patterns(output_dir)

        # Compile results
        results = {
            'distribution_analysis': distribution_analysis,
            'correlation_analysis': correlation_analysis,
            'band_analysis': band_analysis,
            'usage_analysis': usage_analysis
        }

        # Save numerical results
        pd.DataFrame(results).to_csv(f"{output_dir}/analysis_results.csv")

        return results


def analyze_risk_data(input_path: str, output_dir: str) -> Dict:
    """
    Main function to analyze risk data

    Args:
        input_path: Path to merged customer risk data
        output_dir: Directory to save analysis outputs

    Returns:
        Dictionary containing analysis results
    """
    analyzer = RiskAnalysisFramework(input_path)
    results = analyzer.generate_comprehensive_analysis(output_dir)

    print("\nAnalysis Complete!")
    print(f"Results saved to: {output_dir}")

    return results


if __name__ == "__main__":
    # Example usage
    input_path = "output/customer_data_risk.csv"
    output_dir = "output/analysis_results"

    results = analyze_risk_data(input_path, output_dir)