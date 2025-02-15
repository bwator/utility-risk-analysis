import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from delinquency_predictor import DelinquencyPredictor
from dash import dash_table
from sklearn.metrics import classification_report

# Color schemes and styling constants
THEME = {
    'background': '#f8f9fa',
    'card_background': '#ffffff',
    'primary': '#2c3e50',
    'secondary': '#3498db',
    'accent': '#e74c3c',
    'text': '#2c3e50',
    'light_text': '#7f8c8d',
    'success': '#2ecc71',
    'warning': '#f1c40f',
    'danger': '#e74c3c'
}

RISK_COLORS = {
    'Minimal': '#2ecc71',
    'Low': '#3498db',
    'Moderate': '#f1c40f',
    'High': '#e67e22',
    'Severe': '#e74c3c'
}


class EnhancedRiskDashboard:

    def __init__(self, data_path: str):
        """Initialize dashboard with enhanced features"""
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)

        # Initialize predictor
        self.predictor = DelinquencyPredictor()
        self._train_predictor()

        self._preprocess_data()
        self.setup_layout()
        self.setup_callbacks()

    def _train_predictor(self):
        """Train the delinquency predictor"""
        evaluation = self.predictor.train_model(self.df)
        self.model_evaluation = evaluation
        self.predictions = self.predictor.predict_delinquency(self.df)

        # Merge predictions with original data
        self.df = self.df.merge(
            self.predictions,
            on='account_number',
            how='left'
        )

    def _preprocess_data(self):
        """Prepare data with additional metrics"""
        print("Available columns:", list(self.df.columns))
        # Create risk bands
        self.df['risk_band'] = pd.qcut(
            self.df['risk_score'],
            q=5,
            labels=['Minimal', 'Low', 'Moderate', 'High', 'Severe']
        )

        # Try to load intervention data from output directory
        try:
            import os
            # Get the directory from the data_path stored during initialization
            data_dir = os.path.dirname(self.data_path)
            intervention_files = [f for f in os.listdir(data_dir) if 'intervention_data' in f]
            if intervention_files:
                intervention_path = os.path.join(data_dir, intervention_files[-1])
                self.intervention_data = pd.read_csv(intervention_path)

                # Process timeline data
                self.timeline_data = self.intervention_data.groupby('intervention_date').agg({
                    'success': 'mean',
                    'cost': 'sum',
                    'customer_id': 'count'  # Count of interventions per day
                }).reset_index()
                self.timeline_data.rename(columns={'customer_id': 'intervention_count'}, inplace=True)

                # Calculate cumulative metrics
                self.timeline_data['cumulative_success'] = self.timeline_data['success'].cumsum()

                # Process intervention effectiveness
                self.intervention_effectiveness = self.intervention_data.groupby(
                    'intervention_type'
                ).agg({
                    'success': 'mean',
                    'days_to_resolution': 'mean',
                    'cost': 'mean'
                }).reset_index()
            else:
                raise FileNotFoundError("No intervention data found in directory")

        except Exception as e:
            print(f"Error loading intervention data: {e}")
            self.intervention_data = pd.DataFrame()
            self.timeline_data = pd.DataFrame()
            self.intervention_effectiveness = pd.DataFrame()

    def setup_layout(self):
        """Configure dashboard layout with predictive analysis tab"""
        self.app.layout = html.Div([
            # Header
            html.H1(
                "Utility Customer Risk Analysis Dashboard",
                style={
                    'textAlign': 'center',
                    'color': THEME['primary'],
                    'padding': '20px'
                }
            ),

            # Tabs for different views
            dcc.Tabs(
                id='dashboard-tabs',
                value='main',
                children=[
                    dcc.Tab(label='Main Analysis', value='main'),
                    dcc.Tab(label='Multi-Factor Analysis', value='multifactor'),
                    dcc.Tab(label='Enhanced Intervention Analysis', value='intervention'),
                    dcc.Tab(label='Temporal Analysis', value='temporal'),
                    dcc.Tab(label='Geographic Analysis', value='geographic'),
                    dcc.Tab(label='Delinquency Analysis', value='delinquency'),
                    dcc.Tab(label='Predictive Analysis', value='predictive')
                ]
            ),

            # Content div for tab display
            html.Div(id='tab-content', className='p-4')
        ])

    def setup_callbacks(self):
        """Set up dashboard callbacks"""

        @self.app.callback(
            Output('tab-content', 'children'),
            Input('dashboard-tabs', 'value')
        )
        def render_content(tab):
            if tab == 'main':
                return self.create_main_view()
            elif tab == 'multifactor':
                return self.create_multifactor_view()
            elif tab == 'intervention':
                return self.create_intervention_view()
            elif tab == 'temporal':
                return self.create_temporal_view()
            elif tab == 'geographic':
                return self.create_geographic_view()
            elif tab == 'delinquency':
                return self.create_delinquency_view()
            elif tab == 'predictive':
                return self.create_predictive_view()
            return self.create_main_view()

    def create_main_view(self):
        """Create main analysis view"""
        return html.Div([
            # Risk Distribution
            html.Div([
                html.H3("Risk Score Distribution",
                        className="text-xl font-bold mb-4"),
                dcc.Graph(
                    figure=px.histogram(
                        self.df,
                        x='risk_score',
                        color='risk_band',
                        title='Risk Score Distribution'
                    )
                )
            ], className="mb-8"),

            # Risk vs Credit Score
            html.Div([
                html.H3("Risk vs Credit Score Analysis",
                        className="text-xl font-bold mb-4"),
                dcc.Graph(
                    figure=px.scatter(
                        self.df,
                        x='credit_score',
                        y='risk_score',
                        color='risk_band',
                        title='Risk Score vs Credit Score'
                    )
                )
            ], className="mb-8")
        ])

    def create_multifactor_view(self):
        """Create multi-factor analysis view"""
        return html.Div([
            html.Div([
                html.H3("Payment and Usage Pattern Analysis"),
                dcc.Graph(
                    figure=px.scatter(
                        self.df,
                        x='payment_history_score',
                        y='peak_compliance',
                        color='risk_band',
                        size='average_monthly_bill',
                        title='Payment History vs Peak Usage Compliance'
                    )
                )
            ])
        ])

    def create_intervention_view(self):
        """Create enhanced intervention effectiveness analysis view"""
        try:
            # Create original intervention effectiveness scatter plot
            effectiveness_fig = px.scatter(
                self.df,
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
                title='Payment History and Risk Analysis',
                color_discrete_map=RISK_COLORS
            )
            effectiveness_fig.update_layout(
                height=600,
                template='plotly_white',
                xaxis_title="Payment History Score",
                yaxis_title="Risk Score"
            )

            # Return the layout
            return html.Div([
                # Original Analysis
                html.Div([
                    html.H3("Traditional Risk Analysis",
                            className="text-xl font-bold mb-4"),
                    dcc.Graph(figure=effectiveness_fig)
                ], className="mb-8"),

                # Timeline Analysis
                html.Div([
                    html.H3("Intervention Timeline",
                            className="text-xl font-bold mb-4"),
                    dcc.Graph(figure=self._create_timeline_figure())
                ], className="mb-8") if not self.timeline_data.empty else None,

                # Intervention Effectiveness
                html.Div([
                    html.H3("Intervention Type Performance",
                            className="text-xl font-bold mb-4"),
                    dcc.Graph(figure=self._create_effectiveness_figure())
                ], className="mb-8") if not self.intervention_effectiveness.empty else None,

                # Success Rate Analysis
                html.Div([
                    html.H3("Success Rate Analysis",
                            className="text-xl font-bold mb-4"),
                    dcc.Graph(figure=self._create_success_rate_figure())
                ], className="mb-8") if hasattr(self,
                                                'intervention_data') and not self.intervention_data.empty else None
            ])

        except Exception as e:
            print(f"Error in intervention view: {str(e)}")
            return html.Div([
                html.H3("Error loading intervention analysis. Please check data availability.")
            ])

    def create_temporal_view(self):
        """Create temporal analysis view"""
        try:
            # Create customer tenure quantiles for grouping
            self.df['tenure_group'] = pd.qcut(
                self.df['years_of_service'],
                q=5,
                labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
            )

            return html.Div([
                # Payment History Over Time
                html.Div([
                    html.H3("Payment History Trends",
                            className="text-xl font-bold mb-4"),
                    dcc.Graph(
                        figure=px.scatter(
                            self.df,
                            x='years_of_service',
                            y='payment_history_score',
                            color='risk_band',
                            opacity=0.7,
                            title='Payment History by Customer Tenure',
                            labels={
                                'years_of_service': 'Years of Service',
                                'payment_history_score': 'Payment History Score',
                                'risk_band': 'Risk Band'
                            }
                        ).update_layout(
                            template='plotly_white',
                            height=500
                        )
                    )
                ], className="mb-8"),

                # Risk Score Evolution
                html.Div([
                    html.H3("Risk Score Evolution",
                            className="text-xl font-bold mb-4"),
                    dcc.Graph(
                        figure=px.box(
                            self.df,
                            x='tenure_group',
                            y='risk_score',
                            color='risk_band',
                            title='Risk Score Distribution by Customer Tenure',
                            labels={
                                'tenure_group': 'Customer Tenure Percentile',
                                'risk_score': 'Risk Score',
                                'risk_band': 'Risk Band'
                            }
                        ).update_layout(
                            template='plotly_white',
                            height=500
                        )
                    )
                ], className="mb-8"),

                # Payment Trends Analysis
                html.Div([
                    html.H3("Payment Behavior Analysis",
                            className="text-xl font-bold mb-4"),
                    dcc.Graph(
                        figure=px.scatter(
                            self.df,
                            x='years_of_service',
                            y='average_monthly_bill',
                            size='payment_history_score',
                            color='risk_band',
                            opacity=0.7,
                            title='Payment Behavior by Tenure and Bill Size',
                            labels={
                                'years_of_service': 'Years of Service',
                                'average_monthly_bill': 'Average Monthly Bill ($)',
                                'payment_history_score': 'Payment History Score',
                                'risk_band': 'Risk Band'
                            }
                        ).update_layout(
                            template='plotly_white',
                            height=500
                        )
                    )
                ], className="mb-8")
            ])
        except Exception as e:
            print(f"Error in temporal view: {str(e)}")
            return html.Div([
                html.H3("Error loading temporal analysis. Please check data availability."),
                html.Pre(str(e))
            ])

    def create_geographic_view(self):
        """Create geographic analysis view"""
        return html.Div([
            # Risk Score by ZIP Code
            html.Div([
                html.H3("Geographic Risk Distribution",
                        className="text-xl font-bold mb-4"),
                dcc.Graph(
                    figure=px.box(
                        self.df,
                        x='zip_code',
                        y='risk_score',
                        color='risk_band',
                        title='Risk Score Distribution by ZIP Code'
                    )
                )
            ], className="mb-8"),

            # Regional Delinquency Analysis
            html.Div([
                html.H3("Regional Delinquency Analysis",
                        className="text-xl font-bold mb-4"),
                dcc.Graph(
                    figure=px.scatter(
                        self.df,
                        x='regional_delinquency_rate',
                        y='risk_score',
                        color='risk_band',
                        size='average_monthly_bill',
                        title='Risk Score vs Regional Delinquency Rate'
                    )
                )
            ], className="mb-8")
        ])

    def create_delinquency_view(merged_df):
        # Calculate risk quintiles and statistics
        merged_df['risk_quintile'] = pd.qcut(merged_df['risk_score'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

        quintile_stats = merged_df.groupby('risk_quintile').agg({
            'risk_score': 'mean',
            'late_payments_30': 'mean',
            'late_payments_60': 'mean',
            'late_payments_90': 'mean'
        }).reset_index()

        regional_stats = merged_df.groupby('zip_code').agg({
            'risk_score': 'mean',
            'late_payments_30': 'mean',
            'payment_history_score': 'mean'
        }).reset_index()

        # Risk Quintile Analysis
        st.header("Risk Quintile Analysis")
        fig_quintile = px.bar(
            quintile_stats,
            x='risk_quintile',
            y=['late_payments_30', 'late_payments_60', 'late_payments_90'],
            title='Delinquency Patterns by Risk Quintile',
            barmode='group'
        )
        st.plotly_chart(fig_quintile, use_container_width=True)

        # Delinquency Score Distribution
        st.header("Delinquency Risk Distribution")
        fig_risk_dist = px.histogram(
            merged_df,
            x='risk_score',
            color='risk_band',
            title='Risk Score Distribution with Delinquency Bands',
            marginal='box'
        )
        st.plotly_chart(fig_risk_dist, use_container_width=True)

        # Geographic Delinquency Patterns
        st.header("Geographic Delinquency Patterns")
        fig_geo_delinq = px.scatter(
            regional_stats,
            x='payment_history_score',
            y='late_payments_30',
            size='risk_score',
            hover_data=['zip_code'],
            title='Payment History vs Delinquency by ZIP Code'
        )
        st.plotly_chart(fig_geo_delinq, use_container_width=True)

        # Key Insights - Using Streamlit native components
        st.header("Key Delinquency Insights")

        st.subheader("Risk Quintile Analysis")
        high_risk = quintile_stats[quintile_stats['risk_quintile'] == 'Q5']['late_payments_30'].values[0]
        low_risk = quintile_stats[quintile_stats['risk_quintile'] == 'Q1']['late_payments_30'].values[0]
        st.write(f"High-risk quintile averages {high_risk:.1f} late payments per month")
        st.write(f"Low-risk quintile averages {low_risk:.1f} late payments per month")

        st.subheader("Geographic Patterns")
        st.write("ZIP codes with higher risk scores show increased delinquency rates")
        st.write("Payment history strongly correlates with delinquency risk")

        st.subheader("Recommendations")
        st.write("Focus intervention strategies on high-risk quintiles")
        st.write("Develop targeted programs for ZIP codes with elevated risk profiles")
        st.write("Implement early warning system based on payment history patterns")

    def create_predictive_view(self):
        """Create predictive analysis view"""
        return html.Div([
            # Model Performance Metrics
            html.Div([
                html.H3("Model Performance Metrics",
                        className="text-xl font-bold mb-4"),
                html.Div([
                    html.Div([
                        html.H4("Classification Report", className="font-bold"),
                        html.Pre(
                            str(classification_report(
                                self.df['is_delinquent'],
                                (self.df['delinquency_probability'] > 0.5).astype(int)
                            ))
                        )
                    ], className="bg-white p-4 rounded-lg shadow mb-4"),

                    # Feature Importance Plot
                    dcc.Graph(
                        figure=px.bar(
                            self.model_evaluation['feature_importance'],
                            x='importance',
                            y='feature',
                            orientation='h',
                            title='Feature Importance in Predicting Delinquency'
                        )
                    )
                ])
            ], className="mb-8"),

            # Prediction Distribution
            html.Div([
                html.H3("Delinquency Probability Distribution",
                        className="text-xl font-bold mb-4"),
                dcc.Graph(
                    figure=px.histogram(
                        self.df,
                        x='delinquency_probability',
                        color='risk_level',
                        title='Distribution of Delinquency Probabilities',
                        marginal='box'
                    )
                )
            ], className="mb-8"),

            # Risk Factor Analysis
            html.Div([
                html.H3("Risk Factor Analysis",
                        className="text-xl font-bold mb-4"),
                dcc.Graph(
                    figure=px.scatter(
                        self.df,
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
                )
            ], className="mb-8"),

            # High-Risk Customer Table
            html.Div([
                html.H3("High-Risk Customers (Top 10)",
                        className="text-xl font-bold mb-4"),
                dash_table.DataTable(
                    data=self.df.nlargest(10, 'delinquency_probability').to_dict('records'),
                    columns=[
                        {'name': 'Account Number', 'id': 'account_number'},
                        {'name': 'Delinquency Probability', 'id': 'delinquency_probability'},
                        {'name': 'Risk Level', 'id': 'risk_level'},
                        {'name': 'Credit Score', 'id': 'credit_score'},
                        {'name': 'Payment History Score', 'id': 'payment_history_score'}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                )
            ], className="mb-8")
        ])

    def _create_timeline_figure(self):
        """Create intervention timeline visualization"""
        if not hasattr(self, 'timeline_data') or self.timeline_data.empty:
            return go.Figure()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Success Rate
        fig.add_trace(
            go.Scatter(
                x=self.timeline_data['intervention_date'],
                y=self.timeline_data['success'],
                name="Success Rate",
                line=dict(color=THEME['primary'])
            ),
            secondary_y=False
        )

        # Cumulative Success
        fig.add_trace(
            go.Scatter(
                x=self.timeline_data['intervention_date'],
                y=self.timeline_data['cumulative_success'],
                name="Cumulative Success",
                line=dict(color=THEME['success'])
            ),
            secondary_y=True
        )

        fig.update_layout(
            title="Intervention Success Over Time",
            xaxis_title="Date",
            yaxis_title="Success Rate",
            yaxis2_title="Cumulative Success",
            height=400,
            template='plotly_white'
        )

        return fig

    def _create_effectiveness_figure(self):
        """Create intervention effectiveness visualization"""
        if not hasattr(self, 'intervention_effectiveness') or self.intervention_effectiveness.empty:
            return go.Figure()

        fig = go.Figure(data=[
            go.Bar(
                x=self.intervention_effectiveness['intervention_type'],
                y=self.intervention_effectiveness['success'],
                name="Success Rate",
                marker_color=THEME['primary']
            ),
            go.Bar(
                x=self.intervention_effectiveness['intervention_type'],
                y=self.intervention_effectiveness['days_to_resolution'],
                name="Days to Resolution",
                marker_color=THEME['secondary']
            )
        ])

        fig.update_layout(
            title="Intervention Type Performance",
            barmode='group',
            xaxis_title="Intervention Type",
            yaxis_title="Metric Value",
            height=400,
            template='plotly_white'
        )

        return fig

    def _create_success_rate_figure(self):
        """Create success rate analysis visualization"""
        if not hasattr(self, 'intervention_data') or self.intervention_data.empty:
            return go.Figure()

        success_by_type = self.intervention_data.pivot_table(
            values='success',
            index='intervention_type',
            columns='intervention_sequence',
            aggfunc='mean'
        ).fillna(0)

        fig = px.imshow(
            success_by_type,
            title="Success Rate by Intervention Type and Sequence",
            color_continuous_scale="RdYlBu",
            aspect="auto"
        )

        fig.update_layout(
            xaxis_title="Intervention Sequence",
            yaxis_title="Intervention Type",
            height=400,
            template='plotly_white'
        )

        return fig

    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port)