#!/usr/bin/env python
"""
BVMT Stock Market Analysis Pipeline

This script serves as the main entry point for running the entire pipeline,
from scraping data to generating visualizations, training models, and producing insights.

Usage:
    python main.py --all                         # Run the full pipeline
    python main.py --scrape --preprocess         # Only scrape and preprocess data
    python main.py --dashboard                   # Only launch the dashboard
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Import custom modules
from src.scraper import BVMTScraper, scrape_historical
from src.preprocessing import StockDataPreprocessor
from src.visualization import StockVisualizer
from src.model import StockPredictor
# Replace OpenAI LLM import with our local LLM module
from src.llm_local import LocalLLMAnalyst
# Don't import app directly to avoid loading it when not needed
# from src.app import app

def run_scraping(args):
    """Run the data scraping step"""
    print("Starting data scraping...")
    
    try:
        # Try to scrape real data first
        scraper = BVMTScraper()
        companies_data = scraper.scrape(args.output)
        
        # If we want historical data and real scraping worked
        if args.generate_history and companies_data is not None:
            print("Generating historical data...")
            scrape_historical(days=365, output_file=args.output)
            print(f"Scraped data saved to {args.output}")
            return companies_data
    except Exception as e:
        print(f"Error while scraping real data: {str(e)}")
        companies_data = None
    
    # If real scraping failed or returned no data, generate sample data
    if companies_data is None:
        print("Using sample data generator instead of real scraping...")
        sample_data = generate_sample_data(args.output, days=365 if args.generate_history else 30)
        print(f"Sample data saved to {args.output}")
        return sample_data

def run_preprocessing(args):
    """Run the data preprocessing step"""
    print("Starting data preprocessing...")
    
    # Load data
    if os.path.exists(args.input):
        df = pd.read_csv(args.input)
    else:
        print(f"Input file {args.input} not found. Scraping data first...")
        df = run_scraping(args)
    
    # Create preprocessor
    preprocessor = StockDataPreprocessor(args.input)
    
    # Preprocess data
    processed_df = preprocessor.process(args.processed_output)
    
    print(f"Preprocessed data saved to {args.processed_output}")
    
    return processed_df

def run_visualization(args, companies=None):
    """Run the visualization generation step"""
    print("Starting visualization generation...")
    
    # Load preprocessed data
    if os.path.exists(args.processed_output):
        df = pd.read_csv(args.processed_output)
        df['date'] = pd.to_datetime(df['date'])
    else:
        print(f"Preprocessed file {args.processed_output} not found. Preprocessing data first...")
        df = run_preprocessing(args)
    
    # Create visualizer
    visualizer = StockVisualizer(args.processed_output)
    
    # Create plots directory
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # Generate visualizations
    if companies and len(companies) > 0:
        # Generate visualizations for specific companies
        for company in companies:
            if company in df['company'].unique():
                print(f"Generating visualizations for {company}...")
                # Price and volume chart
                fig = visualizer.plot_stock_price(company)
                fig.write_html(f"{args.plots_dir}/{company}_price_volume.html")
                
                # Technical indicators - this is included in the previous plot
                # so no need for a separate plot
                
                # Generate company report if the method exists
                try:
                    report = visualizer.create_company_report(company)
                    with open(f"{args.plots_dir}/{company}_report.html", "w") as f:
                        f.write(report)
                except AttributeError:
                    print("Company report generation not supported in this version")
    else:
        # Generate general market visualizations
        print("Generating market overview visualizations...")
        
        # Sector heatmap
        fig = visualizer.plot_sector_heatmap()
        fig.write_html(f"{args.plots_dir}/sector_heatmap.html")
        
        # Sector volume
        fig = visualizer.plot_sector_volume()
        fig.write_html(f"{args.plots_dir}/sector_volume.html")
        
        # Volume vs Price
        fig = visualizer.plot_volume_vs_price()
        fig.write_html(f"{args.plots_dir}/volume_vs_price.html")
    
    print(f"Visualizations saved to {args.plots_dir}")

def run_models(args, companies=None):
    """Run the model training and prediction step"""
    print("Starting model training and prediction...")
    
    # Load preprocessed data
    if os.path.exists(args.processed_output):
        df = pd.read_csv(args.processed_output)
        df['date'] = pd.to_datetime(df['date'])
    else:
        print(f"Preprocessed file {args.processed_output} not found. Preprocessing data first...")
        df = run_preprocessing(args)
    
    # Create models directory
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Create predictor with the processed data file
    predictor = StockPredictor(args.processed_output)
    
    # Determine which companies to analyze
    if not companies or len(companies) == 0:
        # Select top companies by trading volume
        volume_by_company = df.groupby('company')['volume'].sum()
        companies = volume_by_company.sort_values(ascending=False).head(5).index.tolist()
    
    # Train models for each company
    for company in companies:
        if company in df['company'].unique():
            print(f"Training models for {company}...")
            
            # Train linear regression model
            print(f"Training linear regression model for {company}...")
            lr_model, lr_metrics, lr_results = predictor.train_model(company, model_type='linear')
            
            # Save linear regression model predictions plot
            lr_plot_file = f"{args.models_dir}/{company}_linear_predictions.html"
            predictor.plot_predictions(company, lr_results, lr_plot_file)
            
            # Train random forest model
            print(f"Training random forest model for {company}...")
            rf_model, rf_metrics, rf_results = predictor.train_model(company, model_type='rf')
            
            # Save random forest model predictions plot
            rf_plot_file = f"{args.models_dir}/{company}_rf_predictions.html"
            predictor.plot_predictions(company, rf_results, rf_plot_file)
            
            # Generate feature importance plot
            importance_file = f"{args.models_dir}/{company}_feature_importance.html"
            predictor.feature_importance(company, output_file=importance_file)
            
            # Get next day prediction
            next_day = predictor.predict_next_day(company)
            print(f"\nPrediction for next trading day for {company}:")
            print(f"Current Price: {next_day['current_price']:.3f} TND")
            print(f"Predicted Price: {next_day['predicted_price']:.3f} TND")
            print(f"Expected Change: {next_day['price_change']:.3f} TND ({next_day['price_change_pct']:.2f}%)")
            
            # Save models
            predictor.save_model(company, model_type='linear', output_dir=args.models_dir)
            predictor.save_model(company, model_type='rf', output_dir=args.models_dir)
    
    print(f"Models and predictions saved to {args.models_dir}")

def run_llm_analysis(args, companies=None):
    """Run the LLM-powered analysis step using local models"""
    print("Starting LLM-powered analysis...")
    
    # Check if processed data exists
    if not os.path.exists(args.processed_output):
        print(f"Preprocessed file {args.processed_output} not found. Preprocessing data first...")
        run_preprocessing(args)
    
    # Create insights directory
    os.makedirs(args.insights_dir, exist_ok=True)
    
    # Create local LLM analyst
    print("Initializing LocalLLMAnalyst...")
    try:
        analyst = LocalLLMAnalyst(data_file=args.processed_output)
        print("LocalLLMAnalyst initialized successfully")
        
        # Generate market summary
        print("Generating market summary...")
        try:
            summary = analyst.summarize_market_activity(
                days=7,
                output_file=f"{args.insights_dir}/market_summary.txt"
            )
            print(f"Market summary generated and saved to {args.insights_dir}/market_summary.txt")
            print(f"Summary preview: {summary[:200]}...")
        except Exception as e:
            print(f"Error generating market summary: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Generate company analysis
        print("Generating company analysis...")
        try:
            if companies and len(companies) > 0:
                analysis = analyst.explain_price_movements(
                    companies=companies,
                    days=30,
                    output_file=f"{args.insights_dir}/company_analysis.txt"
                )
            else:
                analysis = analyst.explain_price_movements(
                    days=30,
                    output_file=f"{args.insights_dir}/company_analysis.txt"
                )
            print(f"Company analysis generated and saved to {args.insights_dir}/company_analysis.txt")
            print(f"Analysis preview: {analysis[:200]}...")
        except Exception as e:
            print(f"Error generating company analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Generate sector classifications for example companies
        print("Generating sector classifications...")
        try:
            descriptions = [
                "SFBT is a major beverage company in Tunisia, producing beer and soft drinks.",
                "BIAT is one of the largest private banks in Tunisia offering retail and commercial banking services.",
                "Tunisie Leasing provides equipment and real estate leasing services to businesses."
            ]
            classification = analyst.classify_companies_by_sector(
                company_descriptions=descriptions,
                output_file=f"{args.insights_dir}/sector_classification.txt"
            )
            print(f"Sector classifications generated and saved to {args.insights_dir}/sector_classification.txt")
            print(f"Classification preview: {classification[:200]}...")
        except Exception as e:
            print(f"Error generating sector classifications: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Generate general dataset insights
        print("Generating dataset insights...")
        try:
            insights = analyst.generate_dataframe_insights(
                sample_size=5,
                output_file=f"{args.insights_dir}/data_insights.txt"
            )
            print(f"Dataset insights generated and saved to {args.insights_dir}/data_insights.txt")
            print(f"Insights preview: {insights[:200]}...")
        except Exception as e:
            print(f"Error generating dataset insights: {str(e)}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error initializing LocalLLMAnalyst: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always unload the model when done to free up memory
        try:
            if 'analyst' in locals() and analyst is not None:
                print("Unloading model...")
                analyst.unload_model()
                print("Model unloaded successfully")
        except Exception as e:
            print(f"Error unloading model: {str(e)}")
    
    print(f"LLM insights saved to {args.insights_dir}")

def run_dashboard(args):
    """Launch the dashboard application"""
    print("Starting dashboard...")
    
    # Check if processed data exists
    if not os.path.exists(args.processed_output):
        print(f"Preprocessed file {args.processed_output} not found. Preprocessing data first...")
        run_preprocessing(args)
    
    # Import app only when needed to avoid loading it early
    from app import app
    
    # Run the app directly
    app.run(debug=args.debug, host='127.0.0.1', port=args.port)

def generate_sample_data(output_file='data/bvmt_stocks.csv', days=30):
    """Generate sample stock data"""
    print(f"Generating {days} days of sample stock data...")
    
    # Define sample companies and sectors
    companies = [
        {"name": "BIAT", "sector": "Banking"},
        {"name": "BH", "sector": "Banking"},
        {"name": "ATB", "sector": "Banking"},
        {"name": "BNA", "sector": "Banking"},
        {"name": "SFBT", "sector": "Food & Beverage"},
        {"name": "DELICE", "sector": "Food & Beverage"},
        {"name": "TUNISAIR", "sector": "Transport"},
        {"name": "TELNET", "sector": "Telecom"},
        {"name": "STAR", "sector": "Insurance"},
        {"name": "SIMPAR", "sector": "Construction"}
    ]
    
    # Generate data
    all_data = []
    end_date = datetime.now()
    
    for company in companies:
        # Generate a random starting price between 5 and 100
        base_price = random.uniform(5, 100)
        
        for day in range(days):
            # Calculate date
            current_date = (end_date - timedelta(days=days-day-1)).strftime('%Y-%m-%d')
            
            # Simulate daily price change (-2% to +2%)
            daily_change_pct = random.uniform(-2.0, 2.0)
            
            # Calculate prices
            if day == 0:
                # First day
                last_price = base_price
                opening_price = last_price * (1 - random.uniform(-0.5, 0.5)/100)
            else:
                # Use previous day's closing price as reference
                prev_price = all_data[-1]['last_price']
                last_price = prev_price * (1 + daily_change_pct/100)
                opening_price = prev_price * (1 + random.uniform(-0.5, 0.5)/100)
            
            # Ensure prices are reasonable
            last_price = round(last_price, 3)
            opening_price = round(opening_price, 3)
            
            # Calculate high and low prices
            high = round(max(last_price, opening_price) * (1 + random.uniform(0, 1.0)/100), 3)
            low = round(min(last_price, opening_price) * (1 - random.uniform(0, 1.0)/100), 3)
            
            # Generate a random volume
            volume = int(random.uniform(1000, 10000))
            
            # Add data point
            all_data.append({
                'date': current_date,
                'company': company['name'],
                'sector': company['sector'],
                'last_price': last_price,
                'opening_price': opening_price,
                'high': high,
                'low': low,
                'volume': volume,
                'variation': daily_change_pct
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Sample data saved to {output_file}")
    
    return df

def main():
    """Main entry point for the BVMT Stock Market Analysis Pipeline"""
    parser = argparse.ArgumentParser(description='BVMT Stock Market Analysis Pipeline')
    
    # Pipeline step flags
    parser.add_argument('--all', action='store_true', help='Run the full pipeline')
    parser.add_argument('--scrape', action='store_true', help='Run the data scraping step')
    parser.add_argument('--preprocess', action='store_true', help='Run the data preprocessing step')
    parser.add_argument('--visualize', action='store_true', help='Run the visualization generation step')
    parser.add_argument('--models', action='store_true', help='Run the model training and prediction step')
    parser.add_argument('--analyze', action='store_true', help='Run the LLM-powered analysis step')
    parser.add_argument('--dashboard', action='store_true', help='Launch the dashboard application')
    
    # Additional options
    parser.add_argument('--generate-history', action='store_true', help='Generate historical data when scraping')
    parser.add_argument('--companies', nargs='+', help='Specific companies to analyze')
    parser.add_argument('--debug', action='store_true', help='Run the dashboard in debug mode')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
    
    # File paths
    parser.add_argument('--input', type=str, default='data/bvmt_stocks.csv',
                       help='Input data file path (default: data/bvmt_stocks.csv)')
    parser.add_argument('--output', type=str, default='data/bvmt_stocks.csv',
                       help='Output data file path (default: data/bvmt_stocks.csv)')
    parser.add_argument('--processed-output', type=str, default='data/bvmt_stocks_processed.csv',
                       help='Processed data file path (default: data/bvmt_stocks_processed.csv)')
    parser.add_argument('--plots-dir', type=str, default='data/plots',
                       help='Directory to save plots (default: data/plots)')
    parser.add_argument('--models-dir', type=str, default='data/models',
                       help='Directory to save models (default: data/models)')
    parser.add_argument('--insights-dir', type=str, default='data/insights',
                       help='Directory to save LLM insights (default: data/insights)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If --all is specified, run all steps
    if args.all:
        args.scrape = True
        args.preprocess = True
        args.visualize = True
        args.models = True
        args.analyze = True
    
    # If no steps are specified, show help
    if not any([args.scrape, args.preprocess, args.visualize, args.models, args.analyze, args.dashboard]):
        parser.print_help()
        return
    
    # Run the requested steps
    if args.scrape:
        run_scraping(args)
    
    if args.preprocess:
        run_preprocessing(args)
    
    if args.visualize:
        run_visualization(args, args.companies)
    
    if args.models:
        run_models(args, args.companies)
    
    if args.analyze:
        run_llm_analysis(args, args.companies)
    
    if args.dashboard:
        run_dashboard(args)

if __name__ == "__main__":
    main()