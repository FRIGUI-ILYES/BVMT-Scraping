#!/usr/bin/env python
import os
import pandas as pd
import argparse
import json
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with your OpenAI API key or set it manually.")
    print("Example .env file content: OPENAI_API_KEY=your-api-key-here")

class FinancialLLMAnalyst:
    def __init__(self, data_file='data/bvmt_stocks_processed.csv', api_key=None):
        self.data_file = data_file
        
        # Set API key if provided
        if api_key:
            openai.api_key = api_key
        elif openai_api_key:
            openai.api_key = openai_api_key
        
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the processed stock data"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        self.df = pd.read_csv(self.data_file)
        
        # Ensure date is in datetime format
        self.df['date'] = pd.to_datetime(self.df['date'])
    
    def get_recent_data(self, days=7):
        """Get data from the last N days"""
        # Find the latest date in the dataset
        latest_date = self.df['date'].max()
        
        # Calculate the start date (N days ago)
        start_date = latest_date - pd.Timedelta(days=days)
        
        # Filter data for the last N days
        recent_data = self.df[self.df['date'] >= start_date].copy()
        
        return recent_data
    
    def get_company_data(self, company, days=30):
        """Get recent data for a specific company"""
        # Find the latest date in the dataset
        latest_date = self.df['date'].max()
        
        # Calculate the start date (N days ago)
        start_date = latest_date - pd.Timedelta(days=days)
        
        # Filter data for the company and date range
        company_data = self.df[(self.df['company'] == company) & 
                                (self.df['date'] >= start_date)].copy()
        
        return company_data
    
    def summarize_market_activity(self, days=7, output_file=None):
        """Generate a summary of recent market activity using LLM"""
        recent_data = self.get_recent_data(days)
        
        if recent_data.empty:
            return "No recent data available for market summary."
        
        # Calculate some statistics to provide context for the LLM
        date_range = f"{recent_data['date'].min().strftime('%Y-%m-%d')} to {recent_data['date'].max().strftime('%Y-%m-%d')}"
        
        # Average price changes by company and sort
        avg_changes = recent_data.groupby('company')['variation'].mean().reset_index()
        top_gainers = avg_changes.sort_values('variation', ascending=False).head(5)
        top_losers = avg_changes.sort_values('variation').head(5)
        
        # Average price changes by sector
        sector_changes = recent_data.groupby('sector')['variation'].mean().reset_index()
        sector_changes = sector_changes.sort_values('variation', ascending=False)
        
        # Total trading volume by company and sector
        volume_by_company = recent_data.groupby('company')['volume'].sum().reset_index()
        top_volume_companies = volume_by_company.sort_values('volume', ascending=False).head(5)
        
        volume_by_sector = recent_data.groupby('sector')['volume'].sum().reset_index()
        volume_by_sector = volume_by_sector.sort_values('volume', ascending=False)
        
        # Convert to string representations for the prompt
        top_gainers_str = '\n'.join([f"- {row['company']}: {row['variation']:.2f}%" for _, row in top_gainers.iterrows()])
        top_losers_str = '\n'.join([f"- {row['company']}: {row['variation']:.2f}%" for _, row in top_losers.iterrows()])
        sector_changes_str = '\n'.join([f"- {row['sector']}: {row['variation']:.2f}%" for _, row in sector_changes.iterrows()])
        top_volume_str = '\n'.join([f"- {row['company']}: {row['volume']:.0f}" for _, row in top_volume_companies.iterrows()])
        sector_volume_str = '\n'.join([f"- {row['sector']}: {row['volume']:.0f}" for _, row in volume_by_sector.iterrows()])
        
        # Create the prompt for the LLM
        prompt = f"""
You are a financial analyst for the Tunisian Stock Exchange (BVMT). Provide a comprehensive summary of market activity for the period {date_range}. Use the following data in your analysis:

Top 5 Gainers (Average % Change):
{top_gainers_str}

Top 5 Losers (Average % Change):
{top_losers_str}

Sector Performance (Average % Change):
{sector_changes_str}

Top 5 Companies by Trading Volume:
{top_volume_str}

Sectors by Trading Volume:
{sector_volume_str}

Your summary should:
1. Highlight key trends and patterns observed during this period
2. Analyze which sectors are performing well or struggling
3. Identify potential factors driving these trends
4. Provide a balanced overview of the Tunisian market
5. Include any notable company-specific developments

Format your analysis as a formal market report with appropriate sections.
"""
        
        # Call the LLM API
        response = self.generate_llm_response(prompt)
        
        # Save to file if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Market summary saved to {output_file}")
        
        return response
    
    def explain_price_movements(self, companies=None, days=7, output_file=None):
        """Generate explanations for price movements of specified companies"""
        if companies is None:
            # If no companies specified, use top performers
            recent_data = self.get_recent_data(days)
            avg_changes = recent_data.groupby('company')['variation'].mean().reset_index()
            companies = avg_changes.sort_values('variation', ascending=False).head(3)['company'].tolist()
        
        company_data_list = []
        
        for company in companies:
            company_data = self.get_company_data(company, days)
            
            if company_data.empty:
                continue
            
            # Calculate some statistics
            start_price = company_data.loc[company_data['date'].idxmin(), 'last_price']
            end_price = company_data.loc[company_data['date'].idxmax(), 'last_price']
            price_change = ((end_price - start_price) / start_price) * 100
            
            total_volume = company_data['volume'].sum()
            avg_volume = company_data['volume'].mean()
            
            # Get the sector
            sector = company_data['sector'].iloc[0]
            
            # Create a dictionary with company data
            company_info = {
                'company': company,
                'sector': sector,
                'start_price': start_price,
                'end_price': end_price,
                'price_change': price_change,
                'total_volume': total_volume,
                'avg_volume': avg_volume,
                'daily_data': []
            }
            
            # Add daily price and volume data
            for _, row in company_data.sort_values('date').iterrows():
                company_info['daily_data'].append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'price': row['last_price'],
                    'volume': row['volume'],
                    'variation': row['variation']
                })
            
            company_data_list.append(company_info)
        
        if not company_data_list:
            return "No data available for the specified companies."
        
        # Create the prompt for the LLM
        prompt = f"""
You are a financial analyst for the Tunisian Stock Exchange (BVMT). Provide an in-depth analysis explaining the price movements for the following companies over the past {days} days:

"""
        
        for company_info in company_data_list:
            prompt += f"""
{company_info['company']} ({company_info['sector']}):
- Start Price: {company_info['start_price']:.3f} TND
- End Price: {company_info['end_price']:.3f} TND
- Price Change: {company_info['price_change']:.2f}%
- Total Trading Volume: {company_info['total_volume']:.0f}
- Average Daily Volume: {company_info['avg_volume']:.0f}

Daily Data:
"""
            for day_data in company_info['daily_data']:
                prompt += f"- {day_data['date']}: Price: {day_data['price']:.3f} TND, Change: {day_data['variation']:.2f}%, Volume: {day_data['volume']:.0f}\n"
        
        prompt += """
For each company, analyze:
1. Key factors likely driving the price movements
2. Market sentiment and investor behavior
3. Relevant sector trends and how they affect this company
4. Potential technical indicators (price patterns, volume trends)
5. Comparison to sector peers where relevant

Consider factors specific to Tunisia and emerging markets, such as economic conditions, regulations, or regional events that might impact these stocks.

Provide a separate analysis for each company, followed by an overall perspective on what these movements might indicate about the broader market trends.
"""
        
        # Call the LLM API
        response = self.generate_llm_response(prompt)
        
        # Save to file if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Price movement analysis saved to {output_file}")
        
        return response
    
    def classify_companies_by_sector(self, company_descriptions, output_file=None):
        """Classify companies into sectors based on their descriptions"""
        if not company_descriptions:
            return "No company descriptions provided."
        
        # Get existing sectors for reference
        unique_sectors = self.df['sector'].unique().tolist()
        sectors_list = ', '.join(unique_sectors)
        
        # Create the prompt for the LLM
        prompt = f"""
You are a financial sector classification expert for the Tunisian Stock Exchange (BVMT). Your task is to classify the following companies into appropriate sectors based on their descriptions.

The sectors commonly used in the BVMT include: {sectors_list}

For each company, determine the most appropriate sector from the list above. If none of the existing sectors fit well, you may suggest a new sector name.

Company Descriptions:
"""
        
        for i, desc in enumerate(company_descriptions, 1):
            prompt += f"{i}. {desc}\n"
        
        prompt += """
For each company, provide:
1. The company name extracted from the description
2. The most appropriate sector classification
3. A brief justification for your classification (1-2 sentences)

Format your response as:
Company Name: [Name]
Sector: [Sector]
Justification: [Brief explanation]

Repeat this structure for each company.
"""
        
        # Call the LLM API
        response = self.generate_llm_response(prompt)
        
        # Save to file if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Sector classifications saved to {output_file}")
        
        return response
    
    def generate_dataframe_insights(self, sample_size=10, output_file=None):
        """Generate natural language insights based on a sample of the dataframe"""
        if self.df.empty:
            return "No data available for analysis."
        
        # Get a sample of the data
        if len(self.df) > sample_size:
            sample_df = self.df.sample(sample_size, random_state=42)
        else:
            sample_df = self.df
        
        # Convert sample to a readable format
        sample_str = sample_df.to_string()
        
        # Calculate overall statistics
        stats = {
            'total_companies': self.df['company'].nunique(),
            'total_sectors': self.df['sector'].nunique(),
            'date_range': f"{self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}",
            'avg_price': self.df['last_price'].mean(),
            'avg_volume': self.df['volume'].mean(),
            'avg_variation': self.df['variation'].mean()
        }
        
        # Create the prompt for the LLM
        prompt = f"""
You are a financial data analyst for the Tunisian Stock Exchange (BVMT). Generate insights based on this stock market dataset.

Dataset Summary:
- Total unique companies: {stats['total_companies']}
- Total sectors: {stats['total_sectors']}
- Date range: {stats['date_range']}
- Average price: {stats['avg_price']:.2f} TND
- Average trading volume: {stats['avg_volume']:.2f}
- Average daily price variation: {stats['avg_variation']:.2f}%

Sample of the data (10 rows):
{sample_str}

Based on this information, provide:
1. Key insights about the data structure and what it represents
2. Potential analysis approaches that would yield valuable information
3. Suggested visualizations that would help understand market trends
4. Potential correlations or patterns that might exist in this data
5. Investment strategies that could be informed by analyzing this data

Your insights should be thorough and demonstrate deep understanding of financial markets and data analysis best practices.
"""
        
        # Call the LLM API
        response = self.generate_llm_response(prompt)
        
        # Save to file if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Dataframe insights saved to {output_file}")
        
        return response
    
    def generate_llm_response(self, prompt, model="gpt-4-turbo", max_tokens=1500):
        """Generate a response from the LLM API"""
        try:
            if not openai.api_key:
                return "Error: OpenAI API key not set. Please set the OPENAI_API_KEY environment variable."
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in the Tunisian stock market."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating LLM response: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Generate LLM-powered financial insights')
    parser.add_argument('--data', type=str, default='data/bvmt_stocks_processed.csv',
                        help='Processed data CSV file path')
    parser.add_argument('--output_dir', type=str, default='data/insights',
                        help='Output directory for insights')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key (if not set in environment)')
    parser.add_argument('--analysis', type=str, choices=['market', 'companies', 'sectors', 'insights', 'all'],
                        default='all', help='Type of analysis to perform')
    parser.add_argument('--companies', type=str, nargs='+',
                        help='Companies to analyze (for company analysis)')
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days to include in analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    analyst = FinancialLLMAnalyst(args.data, args.api_key)
    
    # Perform requested analyses
    if args.analysis in ['market', 'all']:
        output_file = f"{args.output_dir}/market_summary.txt"
        analyst.summarize_market_activity(args.days, output_file)
    
    if args.analysis in ['companies', 'all']:
        output_file = f"{args.output_dir}/company_analysis.txt"
        analyst.explain_price_movements(args.companies, args.days, output_file)
    
    if args.analysis in ['sectors', 'all']:
        # Example company descriptions for sector classification
        descriptions = [
            "SFBT is a major beverage company in Tunisia, producing beer and soft drinks.",
            "BIAT is one of the largest private banks in Tunisia offering retail and commercial banking services.",
            "Tunisie Leasing provides equipment and real estate leasing services to businesses."
        ]
        output_file = f"{args.output_dir}/sector_classification.txt"
        analyst.classify_companies_by_sector(descriptions, output_file)
    
    if args.analysis in ['insights', 'all']:
        output_file = f"{args.output_dir}/data_insights.txt"
        analyst.generate_dataframe_insights(10, output_file)

if __name__ == "__main__":
    main() 