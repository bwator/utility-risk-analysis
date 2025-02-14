import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib


class DelinquencyPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None

    def prepare_features(self, df):
        """Prepare features for prediction"""
        features = [
            'credit_score', 'payment_history_score', 'peak_compliance',
            'channel_consistency', 'return_frequency', 'years_of_service',
            'debt_to_income', 'utilization_rate', 'average_monthly_bill',
            'consumption_volatility'
        ]

        # Create target variable (delinquent if any late payments)
        df['is_delinquent'] = (
                (df['late_payments_30'] > 0) |
                (df['late_payments_60'] > 0) |
                (df['late_payments_90'] > 0)
        ).astype(int)

        # Handle categorical variables
        df['autopay_binary'] = (df['autopay_status'] == 'enrolled').astype(int)
        df['budget_billing_binary'] = df['budget_billing'].astype(int)
        features.extend(['autopay_binary', 'budget_billing_binary'])

        return df[features], df['is_delinquent']

    def train_model(self, df):
        """Train the predictive model"""
        X, y = self.prepare_features(df)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)

        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        evaluation = {
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': self.feature_importance
        }

        return evaluation

    def predict_delinquency(self, df):
        """Predict delinquency probability for customers"""
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)

        # Get probabilities of delinquency
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        # Create prediction results
        predictions = pd.DataFrame({
            'account_number': df['account_number'],
            'delinquency_probability': probabilities,
            'risk_level': pd.cut(
                probabilities,
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
                labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
            )
        })

        return predictions

    def save_model(self, path):
        """Save the trained model"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance
            }, path)

    def load_model(self, path):
        """Load a trained model"""
        saved_model = joblib.load(path)
        self.model = saved_model['model']
        self.scaler = saved_model['scaler']
        self.feature_importance = saved_model['feature_importance']