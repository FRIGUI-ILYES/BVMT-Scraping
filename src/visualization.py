#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import os
import datetime as dt

class StockVisualizer:
    def __init__(self, data_file='data/bvmt_stocks_processed.csv'):
        self.data_file = data_file
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the processed stock data"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        self.df = pd.read_csv(self.data_file)
        
        # Ensure date is in datetime format
        self.df['date'] = pd.to_datetime(self.df['date'])
    
    def plot_stock_price(self, company, output_file=None):
        """Create a line plot of stock price over time for a specific company"""
        company_data = self.df[self.df['company'] == company].sort_values('date')
        
        if company_data.empty:
            print(f"No data found for company: {company}")
            return None
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=company_data['date'], 
                y=company_data['last_price'],
                name="Price",
                line=dict(color='rgb(0, 100, 180)', width=2)
            ),
            secondary_y=False
        )
        
        # Add SMA line if available
        if 'price_sma_7' in company_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=company_data['date'], 
                    y=company_data['price_sma_7'],
                    name="7-Day SMA",
                    line=dict(color='rgba(255, 165, 0, 0.7)', width=2, dash='dot')
                ),
                secondary_y=False
            )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=company_data['date'], 
                y=company_data['volume'],
                name="Volume",
                marker=dict(color='rgba(0, 150, 0, 0.3)')
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title=f"{company} - Stock Price & Volume Over Time",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white",
            hovermode="x unified"
        )
        
        fig.update_yaxes(title_text="Price (TND)", secondary_y=False)
        fig.update_yaxes(title_text="Volume", secondary_y=True)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            print(f"Stock price plot saved to {output_file}")
        
        return fig
    
    def plot_sector_heatmap(self, output_file=None):
        """Create a heatmap showing price changes by sector over time"""
        # Get the latest date for each sector
        latest_date = self.df['date'].max()
        
        # Calculate the average price change for each sector over time
        # Group by date and sector, calculating the mean variation
        sector_changes = self.df.groupby(['date', 'sector'])['variation'].mean().reset_index()
        
        # Pivot the data for the heatmap
        pivot_df = sector_changes.pivot(index='date', columns='sector', values='variation')
        
        # Forward fill missing values (when a sector has no data for a date)
        pivot_df = pivot_df.fillna(0)
        
        # Create a smoother heatmap by resampling to weekly
        weekly_pivot = pivot_df.resample('W').mean()
        
        # Create the heatmap using plotly
        fig = px.imshow(
            weekly_pivot.T,  # Transpose for better visualization
            labels=dict(x="Date", y="Sector", color="Price Change (%)"),
            x=weekly_pivot.index,
            y=weekly_pivot.columns,
            color_continuous_scale="RdBu_r",  # Red for negative, blue for positive
            zmin=-3,  # Set min/max for better color scaling
            zmax=3,
            aspect="auto"
        )
        
        fig.update_layout(
            title="Weekly Price Change by Sector",
            xaxis_tickangle=-45,
            coloraxis_colorbar=dict(
                title="Price Change (%)",
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                ticksuffix="%"
            ),
            template="plotly_white"
        )
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            print(f"Sector heatmap saved to {output_file}")
        
        return fig
    
    def plot_sector_volume(self, output_file=None):
        """Create a bar chart of average volume traded per sector"""
        # Calculate the average volume by sector
        sector_volumes = self.df.groupby('sector')['volume'].mean().reset_index()
        sector_volumes = sector_volumes.sort_values('volume', ascending=False)
        
        # Create a bar chart
        fig = px.bar(
            sector_volumes,
            x='sector',
            y='volume',
            color='volume',
            color_continuous_scale='Viridis',
            title="Average Trading Volume by Sector",
            labels={"volume": "Average Volume", "sector": "Sector"}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="Sector",
            yaxis_title="Average Volume",
            coloraxis_showscale=False,
            template="plotly_white"
        )
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            print(f"Sector volume plot saved to {output_file}")
        
        return fig
    
    def plot_volume_vs_price(self, output_file=None):
        """Create a scatter plot of volume vs price change by company"""
        # Get the most recent data point for each company
        latest_date = self.df['date'].max()
        latest_data = self.df[self.df['date'] == latest_date]
        
        # Calculate average volume and price change for each company
        company_stats = self.df.groupby('company').agg({
            'volume': 'mean',
            'variation': 'mean',
            'sector': 'first'  # Keep sector information
        }).reset_index()
        
        # Create scatter plot
        fig = px.scatter(
            company_stats,
            x='volume',
            y='variation',
            color='sector',
            size='volume',
            size_max=50,
            hover_name='company',
            title="Volume vs. Price Change by Company",
            labels={"volume": "Average Volume", "variation": "Average Price Change (%)", "sector": "Sector"}
        )
        
        fig.update_layout(
            xaxis_title="Average Volume",
            yaxis_title="Average Price Change (%)",
            yaxis_ticksuffix="%",
            template="plotly_white"
        )
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            print(f"Volume vs Price plot saved to {output_file}")
        
        return fig
    
    def create_company_report(self, company, output_dir='data/reports'):
        """Create a comprehensive visual report for a specific company"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if company exists in data
        if company not in self.df['company'].unique():
            print(f"Company '{company}' not found in data")
            return
        
        # Create plots
        price_plot_file = f"{output_dir}/{company}_price.html"
        self.plot_stock_price(company, price_plot_file)
        
        # Detailed analysis plots
        company_data = self.df[self.df['company'] == company].sort_values('date')
        
        # Plot price with moving average and volatility
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Price line
        fig1.add_trace(
            go.Scatter(
                x=company_data['date'], 
                y=company_data['last_price'],
                name="Price",
                line=dict(color='rgb(0, 100, 180)', width=2)
            ),
            secondary_y=False
        )
        
        # SMA line
        if 'price_sma_7' in company_data.columns:
            fig1.add_trace(
                go.Scatter(
                    x=company_data['date'], 
                    y=company_data['price_sma_7'],
                    name="7-Day SMA",
                    line=dict(color='rgb(255, 165, 0)', width=2, dash='dot')
                ),
                secondary_y=False
            )
        
        # Volatility
        if 'price_std_7' in company_data.columns:
            fig1.add_trace(
                go.Scatter(
                    x=company_data['date'], 
                    y=company_data['price_std_7'],
                    name="7-Day Volatility",
                    line=dict(color='rgb(255, 0, 0)', width=2)
                ),
                secondary_y=True
            )
        
        fig1.update_layout(
            title=f"{company} - Price and Volatility",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white"
        )
        
        fig1.update_yaxes(title_text="Price (TND)", secondary_y=False)
        fig1.update_yaxes(title_text="Volatility", secondary_y=True)
        
        volatility_file = f"{output_dir}/{company}_volatility.html"
        fig1.write_html(volatility_file)
        
        # Candle chart for short-term analysis
        recent_data = company_data.tail(30)  # Last 30 days
        
        fig2 = go.Figure(data=[go.Candlestick(
            x=recent_data['date'],
            open=recent_data['opening_price'],
            high=recent_data['high'],
            low=recent_data['low'],
            close=recent_data['last_price'],
            name="Price"
        )])
        
        fig2.update_layout(
            title=f"{company} - Recent Price Action (Candlestick)",
            xaxis_title="Date",
            yaxis_title="Price (TND)",
            template="plotly_white"
        )
        
        candle_file = f"{output_dir}/{company}_candle.html"
        fig2.write_html(candle_file)
        
        print(f"Company report for {company} saved to {output_dir}")
    
    def plot_correlations(self, companies, output_file=None):
        """Plot correlation between multiple companies' prices"""
        # Filter for the companies
        filtered_df = self.df[self.df['company'].isin(companies)]
        
        # Create a pivot table of dates and companies with prices
        pivot_df = filtered_df.pivot_table(index='date', columns='company', values='last_price')
        
        # Calculate correlation matrix
        corr_matrix = pivot_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect="auto",
            title="Price Correlation Between Companies"
        )
        
        fig.update_layout(
            xaxis_title="Company",
            yaxis_title="Company",
            template="plotly_white"
        )
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            print(f"Correlation plot saved to {output_file}")
        
        return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize stock data')
    parser.add_argument('--data', type=str, default='data/bvmt_stocks_processed.csv',
                        help='Processed data CSV file path')
    parser.add_argument('--company', type=str, default=None,
                        help='Company to visualize (e.g., BIAT)')
    parser.add_argument('--output_dir', type=str, default='data/plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    visualizer = StockVisualizer(args.data)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.company:
        # Create plots for a specific company
        visualizer.plot_stock_price(args.company, f"{args.output_dir}/{args.company}_price.html")
        visualizer.create_company_report(args.company, args.output_dir)
    else:
        # Create general market plots
        visualizer.plot_sector_heatmap(f"{args.output_dir}/sector_heatmap.html")
        visualizer.plot_sector_volume(f"{args.output_dir}/sector_volume.html")
        visualizer.plot_volume_vs_price(f"{args.output_dir}/volume_vs_price.html")
        
        # Plot some key companies (example)
        key_companies = ['BIAT', 'SFBT', 'TUNISIE LEASING']
        key_companies = [c for c in key_companies if c in visualizer.df['company'].unique()]
        
        for company in key_companies:
            visualizer.plot_stock_price(company, f"{args.output_dir}/{company}_price.html")
        
        if len(key_companies) > 1:
            visualizer.plot_correlations(key_companies, f"{args.output_dir}/company_correlations.html")

if __name__ == "__main__":
    main() 