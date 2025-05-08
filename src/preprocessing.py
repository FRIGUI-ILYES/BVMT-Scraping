#!/usr/bin/env python
import pandas as pd
import numpy as np
import argparse
import os

class StockDataPreprocessor:
    def __init__(self, input_file='data/bvmt_stocks.csv'):
        self.input_file = input_file
        
    def load_data(self):
        """Load data from CSV file"""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        return pd.read_csv(self.input_file)
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Ensure date is in datetime format
        cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
        
        # Sort by date and company
        cleaned_df.sort_values(['date', 'company'], inplace=True)
        
        # Fill missing numeric values with interpolation for each company
        numeric_cols = ['last_price', 'opening_price', 'high', 'low', 'volume', 'variation']
        
        # Group by company and interpolate missing values
        for company in cleaned_df['company'].unique():
            company_mask = cleaned_df['company'] == company
            cleaned_df.loc[company_mask, numeric_cols] = cleaned_df.loc[company_mask, numeric_cols].interpolate(method='linear')
        
        # Fill any remaining NaNs with column medians
        for col in numeric_cols:
            col_median = cleaned_df[col].median()
            cleaned_df[col].fillna(col_median, inplace=True)
        
        # Ensure price columns are strictly positive
        for col in ['last_price', 'opening_price', 'high', 'low']:
            cleaned_df[col] = cleaned_df[col].apply(lambda x: max(0.001, x))
        
        # Ensure volume is integer and positive
        cleaned_df['volume'] = cleaned_df['volume'].apply(lambda x: max(0, int(x)))
        
        # Make sure high ≥ opening ≥ low
        cleaned_df['high'] = cleaned_df.apply(lambda row: max(row['high'], row['opening_price'], row['last_price']), axis=1)
        cleaned_df['low'] = cleaned_df.apply(lambda row: min(row['low'], row['opening_price'], row['last_price']), axis=1)
        
        # Ensure company names and sectors are standardized
        cleaned_df['company'] = cleaned_df['company'].str.strip().str.upper()
        cleaned_df['sector'] = cleaned_df['sector'].str.strip().str.title()
        
        return cleaned_df
    
    def add_features(self, df):
        """Add derived features to the dataframe"""
        # Make a copy to avoid modifying the original
        enhanced_df = df.copy()
        
        # Group by company for company-specific calculations
        companies = enhanced_df['company'].unique()
        
        all_enhanced_rows = []
        
        for company in companies:
            company_data = enhanced_df[enhanced_df['company'] == company].copy()
            company_data = company_data.sort_values('date')
            
            # Calculate rolling averages and volatility (7-day window)
            if len(company_data) >= 3:  # Need at least a few days for meaningful calculations
                # 7-day moving average of closing price
                company_data['price_sma_7'] = company_data['last_price'].rolling(window=7, min_periods=1).mean()
                
                # 7-day standard deviation (volatility)
                company_data['price_std_7'] = company_data['last_price'].rolling(window=7, min_periods=1).std()
                
                # Volume moving average
                company_data['volume_sma_7'] = company_data['volume'].rolling(window=7, min_periods=1).mean()
                
                # Calculate price momentum (percent change over 3 days)
                company_data['momentum_3d'] = company_data['last_price'].pct_change(periods=3).fillna(0) * 100
                
                # Calculate price acceleration (change in momentum)
                company_data['price_accel'] = company_data['momentum_3d'].diff().fillna(0)
                
                # Add relative strength to sector (company price / sector average price)
                sector = company_data['sector'].iloc[0]
                sector_companies = enhanced_df[enhanced_df['sector'] == sector]['company'].unique()
                
                for date in company_data['date'].unique():
                    date_mask = enhanced_df['date'] == date
                    sector_mask = enhanced_df['company'].isin(sector_companies) & date_mask
                    
                    if sector_mask.sum() > 0:
                        sector_avg_price = enhanced_df.loc[sector_mask, 'last_price'].mean()
                        company_date_mask = (company_data['company'] == company) & (company_data['date'] == date)
                        
                        if sector_avg_price > 0 and company_date_mask.sum() > 0:
                            company_data.loc[company_date_mask, 'rel_sector_strength'] = (
                                company_data.loc[company_date_mask, 'last_price'].values[0] / sector_avg_price
                            )
            
            # For companies with less data, just fill with reasonable values
            for col in ['price_sma_7', 'price_std_7', 'volume_sma_7', 'momentum_3d', 'price_accel', 'rel_sector_strength']:
                if col not in company_data.columns:
                    if col in ['price_sma_7']:
                        company_data[col] = company_data['last_price']
                    elif col in ['volume_sma_7']:
                        company_data[col] = company_data['volume']
                    else:
                        company_data[col] = 0
            
            all_enhanced_rows.append(company_data)
        
        # Combine all company data back together
        enhanced_df = pd.concat(all_enhanced_rows, ignore_index=True)
        
        # Fill any NaNs created during the process
        numeric_cols = enhanced_df.select_dtypes(include=['number']).columns
        enhanced_df[numeric_cols] = enhanced_df[numeric_cols].fillna(0)
        
        return enhanced_df
    
    def create_lagged_features(self, df, company_name, n_lags=3):
        """Create lagged features for a specific company for time series modeling"""
        # Filter for the specified company
        company_df = df[df['company'] == company_name].copy()
        
        # Sort by date
        company_df.sort_values('date', inplace=True)
        
        # Create lagged features
        features = ['last_price', 'volume', 'variation', 'price_sma_7', 'price_std_7']
        
        for feature in features:
            for lag in range(1, n_lags + 1):
                company_df[f'{feature}_lag_{lag}'] = company_df[feature].shift(lag)
        
        # Drop rows with NaN values (first n_lags rows)
        company_df.dropna(inplace=True)
        
        return company_df
    
    def process(self, output_file=None):
        """Main preprocessing function"""
        # Load the data
        df = self.load_data()
        
        # Clean the data
        cleaned_df = self.clean_data(df)
        
        # Add features
        enhanced_df = self.add_features(cleaned_df)
        
        # Save processed data if output file is specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            enhanced_df.to_csv(output_file, index=False)
            print(f"Processed data saved to {output_file}")
        
        return enhanced_df

def main():
    parser = argparse.ArgumentParser(description='Preprocess stock data')
    parser.add_argument('--input', type=str, default='data/bvmt_stocks.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default='data/bvmt_stocks_processed.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    preprocessor = StockDataPreprocessor(args.input)
    preprocessor.process(args.output)

if __name__ == "__main__":
    main() 