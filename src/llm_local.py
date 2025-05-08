#!/usr/bin/env python
"""
Local LLM inference module for BVMT Stock Market Analysis

This module provides functionality to generate market insights using
local LLM models from HuggingFace, eliminating the need for external API calls.

The module supports different model sizes and provides fallback mechanisms
for systems with limited memory.
"""
import os
import pandas as pd
import argparse
import json
import gc
import logging
from datetime import datetime, timedelta
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from typing import List, Dict, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default model configuration - use GPU-optimized models
DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FALLBACK_MODEL_NAME = "facebook/opt-350m"
TINY_MODEL_NAME = "facebook/opt-125m"

class LocalLLMAnalyst:
    """
    Analyst class that uses local LLMs to generate financial insights
    without requiring external API calls.
    """
    def __init__(
        self, 
        data_file: str = 'data/bvmt_stocks_processed.csv',
        model_name: str = None,
        use_4bit: bool = True,  # Turn on 4-bit quantization for GPU
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device_map: str = "auto"  # Auto device allocation
    ):
        """
        Initialize the LocalLLMAnalyst with the specified model and parameters.
        
        Args:
            data_file: Path to the processed stock data CSV file
            model_name: HuggingFace model name to use (or None for default)
            use_4bit: Whether to use 4-bit quantization to reduce memory usage
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Top-p sampling parameter (lower = more focused)
            device_map: Device mapping strategy ("auto", "cpu", etc.)
        """
        self.data_file = data_file
        self.df = None
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device_map = device_map
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Load data first
        self.load_data()
        
    def load_data(self):
        """Load the processed stock data"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        self.df = pd.read_csv(self.data_file)
        
        # Ensure date is in datetime format
        self.df['date'] = pd.to_datetime(self.df['date'])
    
    def load_model(self):
        """
        Load the LLM model and tokenizer with appropriate optimizations.
        Uses fallback models if primary model fails to load due to memory constraints.
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Set up quantization config for GPU
            if self.use_4bit and torch.cuda.is_available():
                logger.info("Using 4-bit quantization with CUDA")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            else:
                quantization_config = None
                if torch.cuda.is_available():
                    logger.info("Using CUDA without quantization")
                else:
                    logger.info("Using CPU (CUDA not available)")
            
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load the model with quantization if applicable
            model_kwargs = {}
            if self.use_4bit and torch.cuda.is_available():
                model_kwargs["quantization_config"] = quantization_config
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device_map,
                **model_kwargs
            )
            
            # Create text generation pipeline with loaded model and tokenizer
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True
            )
            
            logger.info(f"Successfully loaded model: {self.model_name}")
            return True
        
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            logger.info("Attempting to free memory...")
            self._free_memory()
            
            # Try fallback models
            if self.model_name != FALLBACK_MODEL_NAME:
                logger.info(f"Trying fallback model: {FALLBACK_MODEL_NAME}")
                self.model_name = FALLBACK_MODEL_NAME
                return self.load_model()
            elif self.model_name != TINY_MODEL_NAME:
                logger.info(f"Trying tiny fallback model: {TINY_MODEL_NAME}")
                self.model_name = TINY_MODEL_NAME
                return self.load_model()
            else:
                logger.error("All model loading attempts failed.")
                return False
    
    def _free_memory(self):
        """Free up memory by removing model and tokenizer and calling garbage collector"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def unload_model(self):
        """Unload model from memory"""
        self._free_memory()
        logger.info("Model unloaded from memory.")
    
    def get_recent_data(self, days: int = 7):
        """Get data from the last N days"""
        # Find the latest date in the dataset
        latest_date = self.df['date'].max()
        
        # Calculate the start date (N days ago)
        start_date = latest_date - pd.Timedelta(days=days)
        
        # Filter data for the last N days
        recent_data = self.df[self.df['date'] >= start_date].copy()
        
        return recent_data
    
    def get_company_data(self, company: str, days: int = 30):
        """Get recent data for a specific company"""
        # Find the latest date in the dataset
        latest_date = self.df['date'].max()
        
        # Calculate the start date (N days ago)
        start_date = latest_date - pd.Timedelta(days=days)
        
        # Filter data for the company and date range
        company_data = self.df[(self.df['company'] == company) & 
                              (self.df['date'] >= start_date)].copy()
        
        return company_data
    
    def generate_local_text(self, prompt: str):
        """Generate text from a prompt using the local LLM"""
        if self.pipeline is None:
            success = self.load_model()
            if not success:
                return "ERROR: Failed to load any LLM model. Please check your system requirements."
        
        try:
            # Format the prompt according to the model's expected format
            if "tinyllama" in self.model_name.lower():
                formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
            elif "opt" in self.model_name.lower():
                # OPT models don't need special formatting
                formatted_prompt = f"User: {prompt}\nAssistant:"
            else:
                # Generic chat format
                formatted_prompt = f"USER: {prompt}\nASSISTANT:"
            
            # Generate response using the pipeline
            result = self.pipeline(
                formatted_prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=50256  # Default padding token
            )
            
            generated_text = result[0]['generated_text']
            
            # Extract just the assistant's response based on model format
            if "tinyllama" in self.model_name.lower():
                response = generated_text.split("<|assistant|>")[-1].strip()
            elif "opt" in self.model_name.lower():
                response = generated_text.split("Assistant:")[-1].strip()
            else:
                # Generic format fallback
                response = generated_text.split("ASSISTANT:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return f"Error generating text: {str(e)}"
    
    def summarize_market_activity(self, days: int = 7, output_file: Optional[str] = None):
        """Generate a summary of recent market activity using local LLM"""
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

Format your analysis as a formal market report with appropriate sections.
"""
        
        # Call the local LLM
        response = self.generate_local_text(prompt)
        
        # Save to file if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            logger.info(f"Market summary saved to {output_file}")
        
        return response
    
    def explain_price_movements(self, companies: Optional[List[str]] = None, days: int = 7, output_file: Optional[str] = None):
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
            
            # Add daily price and volume data (limit to 5 days to keep prompt shorter)
            for _, row in company_data.sort_values('date', ascending=False).head(5).iterrows():
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
You are a financial analyst for the Tunisian Stock Exchange (BVMT). Provide an analysis explaining the price movements for the following companies over the past {days} days:

"""
        
        for company_info in company_data_list:
            prompt += f"""
{company_info['company']} ({company_info['sector']}):
- Start Price: {company_info['start_price']:.3f} TND
- End Price: {company_info['end_price']:.3f} TND
- Price Change: {company_info['price_change']:.2f}%
- Total Trading Volume: {company_info['total_volume']:.0f}
- Average Daily Volume: {company_info['avg_volume']:.0f}

Recent Data (5 most recent days):
"""
            for day_data in company_info['daily_data']:
                prompt += f"- {day_data['date']}: Price: {day_data['price']:.3f} TND, Change: {day_data['variation']:.2f}%, Volume: {day_data['volume']:.0f}\n"
        
        prompt += """
For each company, analyze:
1. Key factors likely driving the price movements
2. Market sentiment and investor behavior
3. Relevant sector trends and how they affect this company
4. Potential technical indicators (price patterns, volume trends)

Provide a separate analysis for each company, followed by an overall perspective on what these movements might indicate about the broader market trends.
"""
        
        # Call the local LLM
        response = self.generate_local_text(prompt)
        
        # Save to file if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            logger.info(f"Price movement analysis saved to {output_file}")
        
        return response
    
    def classify_companies_by_sector(self, company_descriptions: List[str], output_file: Optional[str] = None):
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
        
        # Call the local LLM
        response = self.generate_local_text(prompt)
        
        # Save to file if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            logger.info(f"Sector classifications saved to {output_file}")
        
        return response
    
    def generate_dataframe_insights(self, sample_size: int = 5, output_file: Optional[str] = None):
        """Generate natural language insights based on a sample of the dataframe"""
        if self.df.empty:
            return "No data available for analysis."
        
        # Get a sample of the data (limit sample size to keep prompt shorter)
        actual_sample_size = min(sample_size, 5)
        if len(self.df) > actual_sample_size:
            sample_df = self.df.sample(actual_sample_size, random_state=42)
        else:
            sample_df = self.df
        
        # Format the sample data as a more readable table
        sample_rows = []
        for _, row in sample_df.iterrows():
            sample_rows.append(
                f"Date: {row['date'].strftime('%Y-%m-%d')}, "
                f"Company: {row['company']}, "
                f"Sector: {row['sector']}, "
                f"Price: {row['last_price']:.2f}, "
                f"Change: {row['variation']:.2f}%, "
                f"Volume: {row['volume']:.0f}"
            )
        
        sample_str = "\n".join(sample_rows)
        
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

Sample of the data:
{sample_str}

Based on this information, provide:
1. Key insights about the data structure and what it represents
2. Potential analysis approaches that would yield valuable information
3. Suggested visualizations that would help understand market trends
4. Potential correlations or patterns that might exist in this data
5. Investment strategies that could be informed by analyzing this data

Keep your insights concise and focused on the most important points.
"""
        
        # Call the local LLM
        response = self.generate_local_text(prompt)
        
        # Save to file if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)
            logger.info(f"Dataframe insights saved to {output_file}")
        
        return response

def test_model_availability():
    """Test if models are accessible and estimate memory requirements"""
    models_to_test = [
        {"name": DEFAULT_MODEL_NAME, "size": "1.1B", "memory": "~3GB"},
        {"name": FALLBACK_MODEL_NAME, "size": "350m", "memory": "~350m RAM"},
        {"name": TINY_MODEL_NAME, "size": "125m", "memory": "~125m RAM"}
    ]
    
    print("\n=== Testing Model Availability ===")
    
    for model_info in models_to_test:
        model_name = model_info["name"]
        try:
            # Just check if tokenizer is accessible (much faster than loading model)
            print(f"Testing access to {model_name} ({model_info['size']})...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            print(f"✅ {model_name} is accessible (requires {model_info['memory']} RAM/VRAM)")
        except Exception as e:
            print(f"❌ {model_name} is not accessible: {str(e)}")
    
    # Check system resources
    print("\nSystem resources:")
    import psutil
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB")
    else:
        print("No GPU available")

def main():
    parser = argparse.ArgumentParser(description='Generate local LLM-powered financial insights')
    parser.add_argument('--data', type=str, default='data/bvmt_stocks_processed.csv',
                        help='Processed data CSV file path')
    parser.add_argument('--output_dir', type=str, default='data/insights',
                        help='Output directory for insights')
    parser.add_argument('--model', type=str, default=None,
                        help=f'Model to use (default: {DEFAULT_MODEL_NAME})')
    parser.add_argument('--use_8bit', action='store_true',
                        help='Use 8-bit instead of 4-bit quantization (higher quality, more memory)')
    parser.add_argument('--analysis', type=str, choices=['market', 'companies', 'sectors', 'insights', 'test', 'all'],
                        default='all', help='Type of analysis to perform')
    parser.add_argument('--companies', type=str, nargs='+',
                        help='Companies to analyze (for company analysis)')
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days to include in analysis')
    
    args = parser.parse_args()
    
    # Special case for testing model availability
    if args.analysis == 'test':
        test_model_availability()
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create analyst with specified model
    analyst = LocalLLMAnalyst(
        data_file=args.data,
        model_name=args.model,
        use_4bit=not args.use_8bit
    )
    
    try:
        # Perform requested analyses
        if args.analysis in ['market', 'all']:
            print("Generating market summary...")
            output_file = f"{args.output_dir}/market_summary.txt"
            analyst.summarize_market_activity(args.days, output_file)
        
        if args.analysis in ['companies', 'all']:
            print("Generating company analysis...")
            output_file = f"{args.output_dir}/company_analysis.txt"
            analyst.explain_price_movements(args.companies, args.days, output_file)
        
        if args.analysis in ['sectors', 'all']:
            print("Generating sector classifications...")
            # Example company descriptions for sector classification
            descriptions = [
                "SFBT is a major beverage company in Tunisia, producing beer and soft drinks.",
                "BIAT is one of the largest private banks in Tunisia offering retail and commercial banking services.",
                "Tunisie Leasing provides equipment and real estate leasing services to businesses."
            ]
            output_file = f"{args.output_dir}/sector_classification.txt"
            analyst.classify_companies_by_sector(descriptions, output_file)
        
        if args.analysis in ['insights', 'all']:
            print("Generating dataset insights...")
            output_file = f"{args.output_dir}/data_insights.txt"
            analyst.generate_dataframe_insights(5, output_file)
    
    finally:
        # Always unload the model when done
        if analyst.model is not None:
            analyst.unload_model()

if __name__ == "__main__":
    main() 