#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import argparse
import os
import datetime as dt
import joblib

class StockPredictor:
    def __init__(self, data_file='data/bvmt_stocks_processed.csv'):
        self.data_file = data_file
        self.df = None
        self.model = None
        self.scaler = None
        self.features = None
        self.target = None
        
        self.load_data()
    
    def load_data(self):
        """Load the processed stock data"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        self.df = pd.read_csv(self.data_file)
        
        # Ensure date is in datetime format
        self.df['date'] = pd.to_datetime(self.df['date'])
    
    def prepare_features(self, company, target_col='last_price', n_lags=3):
        """Prepare features for regression model for a specific company"""
        # Filter data for the specific company
        company_data = self.df[self.df['company'] == company].copy()
        
        if company_data.empty:
            raise ValueError(f"No data found for company: {company}")
        
        # Sort by date
        company_data.sort_values('date', inplace=True)
        
        # Create lagged features
        features = ['last_price', 'volume', 'variation']
        
        # Add technical indicators if available
        if 'price_sma_7' in company_data.columns:
            features.extend(['price_sma_7', 'price_std_7', 'momentum_3d'])
        
        # Create lag features
        for feature in features:
            for lag in range(1, n_lags + 1):
                company_data[f'{feature}_lag_{lag}'] = company_data[feature].shift(lag)
        
        # Add day of week
        company_data['day_of_week'] = company_data['date'].dt.dayofweek
        
        # Create target variable (next day's price)
        company_data['target_price'] = company_data[target_col].shift(-1)
        
        # Drop rows with NaNs
        company_data.dropna(inplace=True)
        
        # Define feature columns (excluding target and non-feature columns)
        exclude_cols = ['date', 'company', 'sector', 'target_price', target_col]
        feature_cols = [col for col in company_data.columns if col not in exclude_cols]
        
        # Store feature columns for later use
        self.features = feature_cols
        self.target = 'target_price'
        
        # Create X and y
        X = company_data[feature_cols]
        y = company_data['target_price']
        
        return X, y, company_data
    
    def train_model(self, company, model_type='linear', test_size=0.2, n_lags=3):
        """Train a regression model for a specific company"""
        X, y, _ = self.prepare_features(company, n_lags=n_lags)
        
        # Split data into train and test sets
        # For time series data, we should split by time (not random)
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Create and train model
        if model_type == 'linear':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
        elif model_type == 'rf':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X_train, y_train)
        self.model = model
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Training R^2: {train_r2:.4f}")
        print(f"Test R^2: {test_r2:.4f}")
        
        # Return model and metrics
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        return model, metrics, (X_train, X_test, y_train, y_test, y_pred_train, y_pred_test)
    
    def plot_predictions(self, company, model_results, output_file=None):
        """Plot actual vs predicted prices"""
        _, _, train_data = self.prepare_features(company)
        
        X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = model_results
        
        # Create a DataFrame for plotting
        test_size = len(y_test)
        train_size = len(y_train)
        total_size = train_size + test_size
        
        dates = train_data['date'].iloc[-total_size:].values
        
        plot_df = pd.DataFrame({
            'date': dates,
            'actual': np.concatenate([y_train.values, y_test.values]),
            'predicted': np.concatenate([y_pred_train, y_pred_test]),
            'dataset': ['train'] * train_size + ['test'] * test_size
        })
        
        # Create plot
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(
            go.Scatter(
                x=plot_df['date'], 
                y=plot_df['actual'],
                name="Actual Price",
                line=dict(color='rgb(0, 100, 180)', width=2)
            )
        )
        
        # Add predicted prices for training set
        train_df = plot_df[plot_df['dataset'] == 'train']
        fig.add_trace(
            go.Scatter(
                x=train_df['date'], 
                y=train_df['predicted'],
                name="Predicted (Train)",
                line=dict(color='rgba(0, 200, 0, 0.5)', width=2)
            )
        )
        
        # Add predicted prices for test set
        test_df = plot_df[plot_df['dataset'] == 'test']
        fig.add_trace(
            go.Scatter(
                x=test_df['date'], 
                y=test_df['predicted'],
                name="Predicted (Test)",
                line=dict(color='rgba(255, 0, 0, 0.7)', width=2)
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"{company} - Actual vs Predicted Prices",
            xaxis_title="Date",
            yaxis_title="Price (TND)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white"
        )
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            print(f"Prediction plot saved to {output_file}")
        
        return fig
    
    def save_model(self, company, model_type='linear', output_dir='models'):
        """Save the trained model for future use"""
        if self.model is None:
            raise ValueError("No trained model available. Call train_model() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        model_file = f"{output_dir}/{company}_{model_type}_model.joblib"
        feature_file = f"{output_dir}/{company}_{model_type}_features.joblib"
        
        joblib.dump(self.model, model_file)
        joblib.dump({'features': self.features, 'target': self.target}, feature_file)
        
        print(f"Model saved to {model_file}")
        print(f"Feature info saved to {feature_file}")
    
    def load_model(self, company, model_type='linear', input_dir='models'):
        """Load a previously trained model"""
        model_file = f"{input_dir}/{company}_{model_type}_model.joblib"
        feature_file = f"{input_dir}/{company}_{model_type}_features.joblib"
        
        if not os.path.exists(model_file) or not os.path.exists(feature_file):
            raise FileNotFoundError(f"Model or feature files not found for {company}")
        
        self.model = joblib.load(model_file)
        feature_info = joblib.load(feature_file)
        self.features = feature_info['features']
        self.target = feature_info['target']
        
        print(f"Model loaded from {model_file}")
    
    def predict_next_day(self, company):
        """Predict the next day's price for a company using the trained model"""
        if self.model is None:
            raise ValueError("No trained model available. Call train_model() or load_model() first.")
        
        # Get the latest data for the company
        company_data = self.df[self.df['company'] == company].copy()
        company_data.sort_values('date', inplace=True)
        
        # Prepare the latest data point
        latest_data = company_data.iloc[-1:].copy()
        
        # Check if we have all required feature columns
        missing_features = [col for col in self.features if col not in latest_data.columns]
        if missing_features:
            # We need to calculate the missing features
            # This typically happens for lagged features that weren't in the original data
            
            # Create a copy of the dataframe for feature engineering
            temp_df = company_data.copy()
            
            # Create all lag features that might be needed
            for col in ['last_price', 'volume', 'variation', 'price_sma_7', 'price_std_7', 'momentum_3d']:
                if col in temp_df.columns:
                    for lag in range(1, 10):  # Assuming max lag of 10
                        lag_col = f"{col}_lag_{lag}"
                        if lag_col in self.features:
                            temp_df[lag_col] = temp_df[col].shift(lag)
            
            # If day_of_week is a feature
            if 'day_of_week' in self.features and 'day_of_week' not in temp_df.columns:
                temp_df['day_of_week'] = temp_df['date'].dt.dayofweek
            
            # Get the latest complete record with all features
            latest_complete = temp_df.iloc[-1:].copy()
            
            # Extract only the features needed for prediction
            X_pred = latest_complete[self.features]
        else:
            # We already have all the features
            X_pred = latest_data[self.features]
        
        # Make prediction
        predicted_price = self.model.predict(X_pred)[0]
        
        # Get the current price for comparison
        current_price = company_data['last_price'].iloc[-1]
        
        # Calculate expected change
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        result = {
            'company': company,
            'current_date': company_data['date'].iloc[-1],
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct
        }
        
        return result
    
    def feature_importance(self, company, model_type='rf', output_file=None):
        """Calculate and plot feature importance for Random Forest model"""
        if self.model is None or model_type != 'rf':
            # Train a Random Forest model if not already available
            self.train_model(company, model_type='rf')
        
        # Extract the Random Forest regressor from the pipeline
        rf_model = self.model.named_steps['regressor']
        
        # Get feature importances
        importances = rf_model.feature_importances_
        
        # Create a DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Create plot
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title=f"{company} - Feature Importance",
            labels={'importance': 'Importance', 'feature': 'Feature'}
        )
        
        fig.update_layout(
            yaxis=dict(categoryorder='total ascending'),
            template="plotly_white"
        )
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            print(f"Feature importance plot saved to {output_file}")
        
        return fig, importance_df

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate stock price prediction models')
    parser.add_argument('--data', type=str, default='data/bvmt_stocks_processed.csv',
                        help='Processed data CSV file path')
    parser.add_argument('--company', type=str, required=True,
                        help='Company to analyze (e.g., BIAT)')
    parser.add_argument('--model', type=str, default='linear', choices=['linear', 'rf'],
                        help='Model type: linear (Linear Regression) or rf (Random Forest)')
    parser.add_argument('--lags', type=int, default=3,
                        help='Number of lag features to create')
    parser.add_argument('--output_dir', type=str, default='data/models',
                        help='Output directory for models and plots')
    parser.add_argument('--save', action='store_true',
                        help='Save the trained model')
    
    args = parser.parse_args()
    
    predictor = StockPredictor(args.data)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    print(f"Training {args.model} model for {args.company} with {args.lags} lag features...")
    model, metrics, model_results = predictor.train_model(
        args.company, 
        model_type=args.model, 
        n_lags=args.lags
    )
    
    # Plot results
    plot_file = f"{args.output_dir}/{args.company}_{args.model}_predictions.html"
    predictor.plot_predictions(args.company, model_results, plot_file)
    
    # If using Random Forest, show feature importance
    if args.model == 'rf':
        importance_file = f"{args.output_dir}/{args.company}_feature_importance.html"
        predictor.feature_importance(args.company, output_file=importance_file)
    
    # Predict next day's price
    next_day = predictor.predict_next_day(args.company)
    print("\nPrediction for next trading day:")
    print(f"Company: {next_day['company']}")
    print(f"Current Price: {next_day['current_price']:.3f} TND")
    print(f"Predicted Price: {next_day['predicted_price']:.3f} TND")
    print(f"Expected Change: {next_day['price_change']:.3f} TND ({next_day['price_change_pct']:.2f}%)")
    
    # Save model if requested
    if args.save:
        predictor.save_model(args.company, model_type=args.model, output_dir=args.output_dir)

if __name__ == "__main__":
    main() 