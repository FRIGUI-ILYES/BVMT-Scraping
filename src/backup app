#!/usr/bin/env python
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import argparse
import sys
from pathlib import Path

# Add project root to Python path if running directly
if __name__ == "__main__":
    script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_path)  # Go up one level to project root
    sys.path.insert(0, project_root)

# Import custom modules
try:
    # Try importing directly (when run from project root)
    from src.scraper import BVMTScraper, scrape_historical
    from src.preprocessing import StockDataPreprocessor
    from src.visualization import StockVisualizer
    from src.model import StockPredictor
    from src.llm_local import LocalLLMAnalyst
except ImportError:
    # If that fails, try importing without 'src.' prefix (when run from src folder)
    from scraper import BVMTScraper, scrape_historical
    from preprocessing import StockDataPreprocessor
    from visualization import StockVisualizer
    from model import StockPredictor
    from llm_local import LocalLLMAnalyst

# Initialize the Dashboard app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "BVMT Stock Analysis Dashboard"

# Global variables
# Make paths relative to the project root
if __name__ == "__main__":
    DATA_FILE = os.path.join(project_root, 'data/bvmt_stocks.csv')
    PROCESSED_DATA_FILE = os.path.join(project_root, 'data/bvmt_stocks_processed.csv')
else:
    DATA_FILE = 'data/bvmt_stocks.csv'
    PROCESSED_DATA_FILE = 'data/bvmt_stocks_processed.csv'


def load_data():
    """Load and preprocess data if available, otherwise generate sample data"""
    # Make sure to use the correct path regardless of where we're running from
    data_path = DATA_FILE
    processed_path = PROCESSED_DATA_FILE
    
    if not os.path.exists(data_path):
        print(f"No data file found at {data_path}. Generating sample data...")
        scraper = BVMTScraper()
        scraper.scrape(data_path)
        # Generate 30 days of historical data
        scrape_historical(days=30, output_file=data_path)
    
    # Preprocess the data
    if not os.path.exists(processed_path) or \
       os.path.getmtime(data_path) > os.path.getmtime(processed_path):
        print(f"Processing data at {processed_path}...")
        preprocessor = StockDataPreprocessor(data_path)
        preprocessor.process(processed_path)
    
    # Load the processed data
    df = pd.read_csv(processed_path)
    df['date'] = pd.to_datetime(df['date'])
    
    return df


# Load data
df = load_data()

# Get list of companies and sectors
companies = sorted(df['company'].unique())
sectors = sorted(df['sector'].unique())

# Create the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Tunisian Stock Exchange Analysis", className="text-center my-4"),
            html.P("Interactive dashboard for BVMT stock analysis", className="text-center lead")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Select Data Parameters"),
                dbc.CardBody([
                    html.Label("Select Company:"),
                    dcc.Dropdown(
                        id='company-dropdown',
                        options=[{'label': company, 'value': company} for company in companies],
                        value=companies[0] if companies else None,
                        clearable=False
                    ),
                    
                    html.Label("Select Sector:", className="mt-3"),
                    dcc.Dropdown(
                        id='sector-dropdown',
                        options=[{'label': sector, 'value': sector} for sector in sectors],
                        value=None,
                        clearable=True
                    ),
                    
                    html.Label("Date Range:", className="mt-3"),
                    dcc.DatePickerRange(
                        id='date-picker',
                        min_date_allowed=df['date'].min().date(),
                        max_date_allowed=df['date'].max().date(),
                        start_date=df['date'].max() - pd.Timedelta(days=30),
                        end_date=df['date'].max(),
                        className="mb-3"
                    ),
                    
                    html.Label("Analysis Type:", className="mt-3"),
                    dcc.RadioItems(
                        id='analysis-type',
                        options=[
                            {'label': 'Price & Volume', 'value': 'price'},
                            {'label': 'Technical Indicators', 'value': 'technical'},
                            {'label': 'Prediction', 'value': 'prediction'}
                        ],
                        value='price',
                        className="mb-3"
                    ),
                    
                    dbc.Button("Update Analysis", id="update-button", color="primary", className="mt-3")
                ])
            ], className="mb-4")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5(id="chart-title", children="Stock Price Analysis")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-1",
                        type="default",
                        children=dcc.Graph(id='main-chart', figure={})
                    )
                ])
            ], className="mb-4")
        ], width=9)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Market Insights"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            html.Div(id="company-summary", className="mt-3")
                        ], label="Company Summary"),
                        dbc.Tab([
                            dcc.Loading(
                                id="loading-2",
                                type="default",
                                children=dcc.Graph(id='volume-chart', figure={})
                            )
                        ], label="Volume Analysis"),
                        dbc.Tab([
                            dcc.Loading(
                                id="loading-3",
                                type="default",
                                children=dcc.Graph(id='sector-chart', figure={})
                            )
                        ], label="Sector Overview"),
                        dbc.Tab([
                            html.Div([
                                html.H5("Dataset Information", className="mt-3"),
                                html.Div(id="data-structure-info")
                            ])
                        ], label="Data Structure")
                    ])
                ])
            ])
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI-Powered Analysis"),
                dbc.CardBody([
                    html.Div([
                        dbc.Button("Generate Market Insight", id="generate-insight-button", color="success"),
                        dbc.Spinner(html.Div(id="llm-insight", className="mt-3"))
                    ])
                ])
            ], className="my-4")
        ], width=12)
    ]),
    
    html.Footer([
        html.P("BVMT Stock Analysis Dashboard - Created for educational purposes", className="text-center text-muted")
    ], className="mt-5")
    
], fluid=True)


@app.callback(
    [Output('main-chart', 'figure'),
     Output('chart-title', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('company-dropdown', 'value'),
     State('sector-dropdown', 'value'),
     State('date-picker', 'start_date'),
     State('date-picker', 'end_date'),
     State('analysis-type', 'value')]
)
def update_main_chart(n_clicks, company, sector, start_date, end_date, analysis_type):
    """Update the main chart based on user selection"""
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter data by date
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    if sector and not company:
        # If sector is selected but not company, show sector-level chart
        sector_df = filtered_df[filtered_df['sector'] == sector]
        
        if analysis_type == 'price':
            # Create a line chart of average price by date for the sector
            sector_prices = sector_df.groupby('date')['last_price'].mean().reset_index()
            fig = px.line(
                sector_prices, 
                x='date', 
                y='last_price',
                title=f"{sector} Sector - Average Price Trend"
            )
            fig.update_layout(template='plotly_white')
            return fig, f"{sector} Sector - Average Price Trend"
        
        elif analysis_type == 'technical':
            # Show performance comparison of companies in the sector
            companies_in_sector = sector_df['company'].unique()
            performance_df = pd.DataFrame()
            
            for comp in companies_in_sector:
                comp_data = sector_df[sector_df['company'] == comp]
                if not comp_data.empty:
                    start_price = comp_data.loc[comp_data['date'].idxmin(), 'last_price']
                    comp_data = comp_data.copy()
                    comp_data['normalized_price'] = comp_data['last_price'] / start_price * 100
                    comp_data['company_name'] = comp
                    performance_df = pd.concat([performance_df, comp_data[['date', 'company_name', 'normalized_price']]])
            
            fig = px.line(
                performance_df, 
                x='date', 
                y='normalized_price', 
                color='company_name',
                title=f"{sector} Sector - Company Performance Comparison (Normalized to 100)"
            )
            fig.update_layout(yaxis_title="Normalized Price (Base=100)", template='plotly_white')
            return fig, f"{sector} Sector - Company Performance Comparison"
        
        else:  # prediction
            fig = go.Figure()
            fig.add_annotation(
                text="Sector-level prediction not supported. Please select a specific company.",
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(template='plotly_white')
            return fig, "Prediction Analysis"
    
    elif company:
        # Filter for the selected company
        company_df = filtered_df[filtered_df['company'] == company]
        
        if company_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {company} in the selected date range.",
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(template='plotly_white')
            return fig, f"{company} - No Data Available"
        
        if analysis_type == 'price':
            # Create price and volume chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=company_df['date'], 
                    y=company_df['last_price'],
                    name="Price",
                    line=dict(color='rgb(0, 100, 180)', width=2)
                ),
                secondary_y=False
            )
            
            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=company_df['date'], 
                    y=company_df['volume'],
                    name="Volume",
                    marker=dict(color='rgba(0, 150, 0, 0.3)')
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title=f"{company} - Stock Price & Volume",
                xaxis_title="Date",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                template="plotly_white",
                hovermode="x unified"
            )
            
            fig.update_yaxes(title_text="Price (TND)", secondary_y=False)
            fig.update_yaxes(title_text="Volume", secondary_y=True)
            
            return fig, f"{company} - Stock Price & Volume"
        
        elif analysis_type == 'technical':
            # Create technical indicators chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=company_df['date'], 
                    y=company_df['last_price'],
                    name="Price",
                    line=dict(color='rgb(0, 100, 180)', width=2)
                ),
                secondary_y=False
            )
            
            # Add 7-day SMA
            if 'price_sma_7' in company_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=company_df['date'], 
                        y=company_df['price_sma_7'],
                        name="7-Day SMA",
                        line=dict(color='rgba(255, 165, 0, 0.7)', width=2, dash='dot')
                    ),
                    secondary_y=False
                )
            
            # Add volatility
            if 'price_std_7' in company_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=company_df['date'], 
                        y=company_df['price_std_7'],
                        name="7-Day Volatility",
                        line=dict(color='rgba(255, 0, 0, 0.7)', width=2)
                    ),
                    secondary_y=True
                )
            
            # Update layout
            fig.update_layout(
                title=f"{company} - Technical Indicators",
                xaxis_title="Date",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                template="plotly_white",
                hovermode="x unified"
            )
            
            fig.update_yaxes(title_text="Price (TND)", secondary_y=False)
            fig.update_yaxes(title_text="Volatility", secondary_y=True)
            
            return fig, f"{company} - Technical Indicators"
        
        else:  # prediction
            # Try to run a simple linear regression model
            try:
                predictor = StockPredictor(PROCESSED_DATA_FILE)
                model, metrics, model_results = predictor.train_model(company, model_type='linear')
                
                # Unpack model results
                X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = model_results
                
                # Get dates for the prediction (we need to join with the original data)
                _, _, train_data = predictor.prepare_features(company)
                
                # Get appropriate dates
                test_size = len(y_test)
                train_size = len(y_train)
                total_size = train_size + test_size
                
                dates = train_data['date'].iloc[-total_size:].values
                
                # Create the plot DataFrame
                plot_df = pd.DataFrame({
                    'date': dates,
                    'actual': pd.concat([y_train, y_test]).values,
                    'predicted': pd.concat([pd.Series(y_pred_train), pd.Series(y_pred_test)]).values,
                    'dataset': ['train'] * train_size + ['test'] * test_size
                })
                
                # Make a prediction for the next day
                next_day_pred = predictor.predict_next_day(company)
                next_day_date = next_day_pred['current_date'] + pd.Timedelta(days=1)
                
                # Add the prediction to the DataFrame
                plot_df = pd.concat([
                    plot_df,
                    pd.DataFrame({
                        'date': [next_day_date],
                        'actual': [None],
                        'predicted': [next_day_pred['predicted_price']],
                        'dataset': ['forecast']
                    })
                ])
                
                # Create the plot
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
                
                # Add predicted prices for training set (partially transparent)
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
                
                # Add forecast point
                forecast_df = plot_df[plot_df['dataset'] == 'forecast']
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df['date'],
                        y=forecast_df['predicted'],
                        name="Next Day Forecast",
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='star')
                    )
                )
                
                # Add annotation for forecast value
                fig.add_annotation(
                    x=forecast_df['date'].iloc[0],
                    y=forecast_df['predicted'].iloc[0],
                    text=f"Forecast: {forecast_df['predicted'].iloc[0]:.2f} TND",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363",
                    ax=70,
                    ay=-40
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{company} - Price Prediction Model",
                    xaxis_title="Date",
                    yaxis_title="Price (TND)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    template="plotly_white"
                )
                
                return fig, f"{company} - Price Prediction Model"
                
            except Exception as e:
                # If prediction fails, show an error message
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Could not generate prediction: {str(e)}",
                    showarrow=False,
                    font=dict(size=14)
                )
                fig.update_layout(template='plotly_white')
                return fig, f"{company} - Prediction Error"
    
    else:
        # No selections - show general market overview
        # Calculate daily average price across all companies
        market_avg = filtered_df.groupby('date')['last_price'].mean().reset_index()
        
        fig = px.line(
            market_avg,
            x='date',
            y='last_price',
            title="BVMT Market Overview - Average Stock Price"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Price (TND)",
            template="plotly_white"
        )
        
        return fig, "BVMT Market Overview"


@app.callback(
    Output('volume-chart', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('sector-dropdown', 'value'),
     State('date-picker', 'start_date'),
     State('date-picker', 'end_date')]
)
def update_volume_chart(n_clicks, sector, start_date, end_date):
    """Update the volume analysis chart"""
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter data by date
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    if sector:
        # Filter by sector if selected
        filtered_df = filtered_df[filtered_df['sector'] == sector]
    
    # Calculate average volume by company
    volume_by_company = filtered_df.groupby('company')['volume'].mean().reset_index()
    # Sort by volume in descending order and take top 10
    top_companies = volume_by_company.sort_values('volume', ascending=False).head(10)
    
    # Create a bar chart
    fig = px.bar(
        top_companies,
        x='company',
        y='volume',
        color='volume',
        color_continuous_scale='Viridis',
        title="Top 10 Companies by Average Trading Volume"
    )
    
    fig.update_layout(
        xaxis_title="Company",
        yaxis_title="Average Volume",
        template="plotly_white",
        xaxis_tickangle=-45
    )
    
    return fig


@app.callback(
    Output('sector-chart', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('date-picker', 'start_date'),
     State('date-picker', 'end_date')]
)
def update_sector_chart(n_clicks, start_date, end_date):
    """Update the sector overview chart"""
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter data by date
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    # Calculate average variation by sector
    sector_perf = filtered_df.groupby('sector')['variation'].mean().reset_index()
    sector_perf = sector_perf.sort_values('variation', ascending=False)
    
    # Create a horizontal bar chart
    fig = px.bar(
        sector_perf,
        y='sector',
        x='variation',
        orientation='h',
        color='variation',
        color_continuous_scale='RdBu',
        title="Sector Performance (Average % Change)"
    )
    
    fig.update_layout(
        xaxis_title="Average Price Change (%)",
        yaxis_title="Sector",
        template="plotly_white",
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
    )
    
    return fig


@app.callback(
    Output('company-summary', 'children'),
    [Input('company-dropdown', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_company_summary(company, start_date, end_date):
    """Update the company summary text"""
    if not company:
        return html.P("Please select a company to view its summary.")
    
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter data by company and date
    company_df = df[(df['company'] == company) & 
                     (df['date'] >= start_date) & 
                     (df['date'] <= end_date)]
    
    if company_df.empty:
        return html.P(f"No data available for {company} in the selected date range.")
    
    # Calculate statistics
    start_price = company_df.loc[company_df['date'].idxmin(), 'last_price']
    end_price = company_df.loc[company_df['date'].idxmax(), 'last_price']
    price_change = end_price - start_price
    price_change_pct = (price_change / start_price) * 100
    
    avg_volume = company_df['volume'].mean()
    total_volume = company_df['volume'].sum()
    
    high = company_df['high'].max()
    low = company_df['low'].min()
    
    # Get sector
    sector = company_df['sector'].iloc[0]
    
    # Create summary
    summary = [
        html.H5(f"{company} ({sector})"),
        html.P([
            html.Strong("Period: "), 
            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        ]),
        html.Div([
            html.Div([
                html.P([html.Strong("Start Price: "), f"{start_price:.3f} TND"]),
                html.P([html.Strong("End Price: "), f"{end_price:.3f} TND"]),
                html.P([
                    html.Strong("Change: "), 
                    html.Span(
                        f"{price_change:.3f} TND ({price_change_pct:.2f}%)",
                        style={'color': 'green' if price_change_pct >= 0 else 'red'}
                    )
                ]),
            ], className="col-md-6"),
            html.Div([
                html.P([html.Strong("Highest Price: "), f"{high:.3f} TND"]),
                html.P([html.Strong("Lowest Price: "), f"{low:.3f} TND"]),
                html.P([html.Strong("Avg. Daily Volume: "), f"{avg_volume:.0f}"]),
                html.P([html.Strong("Total Volume: "), f"{total_volume:.0f}"]),
            ], className="col-md-6")
        ], className="row")
    ]
    
    return html.Div(summary)


@app.callback(
    Output('llm-insight', 'children'),
    [Input('generate-insight-button', 'n_clicks')],
    [State('company-dropdown', 'value'),
     State('date-picker', 'start_date'),
     State('date-picker', 'end_date')]
)
def generate_llm_insight(n_clicks, company, start_date, end_date):
    """Generate insights without relying on external APIs"""
    if not n_clicks:
        return html.P("Click the button to generate insights.")
    
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter data
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    if company:
        # Company-specific analysis
        company_df = filtered_df[filtered_df['company'] == company]
        
        if company_df.empty:
            return html.P(f"No data available for {company} in the selected date range.")
        
        # Calculate statistics
        start_price = company_df.loc[company_df['date'].idxmin(), 'last_price']
        end_price = company_df.loc[company_df['date'].idxmax(), 'last_price']
        price_change = end_price - start_price
        price_change_pct = (price_change / start_price) * 100
        
        avg_volume = company_df['volume'].mean()
        max_volume_day = company_df.loc[company_df['volume'].idxmax()]
        min_price_day = company_df.loc[company_df['last_price'].idxmin()]
        max_price_day = company_df.loc[company_df['last_price'].idxmax()]
        
        # Generate insight
        sector = company_df['sector'].iloc[0]
        trend = "upward" if price_change > 0 else "downward"
        
        # Format the insight with HTML
        insight = [
            html.H4(f"Analysis for {company}"),
            html.P([
                f"{company} ({sector}) showed an overall {trend} trend during the period from ",
                html.Strong(f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, "),
                f"with a price change of ",
                html.Strong(f"{price_change:.2f} TND ({price_change_pct:.2f}%).")
            ]),
            html.P([
                f"The stock reached its highest price of ",
                html.Strong(f"{max_price_day['last_price']:.2f} TND "),
                f"on {max_price_day['date'].strftime('%Y-%m-%d')}, and its lowest price of ",
                html.Strong(f"{min_price_day['last_price']:.2f} TND "),
                f"on {min_price_day['date'].strftime('%Y-%m-%d')}."
            ]),
            html.P([
                f"Trading activity was notable with an average daily volume of ",
                html.Strong(f"{avg_volume:.0f} shares. "),
                f"The highest trading volume occurred on {max_volume_day['date'].strftime('%Y-%m-%d')} ",
                f"with {max_volume_day['volume']:.0f} shares traded."
            ]),
            html.H5("Interpretation"),
            html.P([
                f"The {'positive' if price_change > 0 else 'negative'} trend suggests ",
                f"{'investor confidence and potential growth opportunities.' if price_change > 0 else 'potential concerns or market corrections.'}",
                f" Investors should consider {'maintaining positions and watching for continued growth.' if price_change > 0 else 'monitoring closely for stabilization or further decline.'}"
            ])
        ]
        
        return html.Div(insight)
    
    else:
        # Market-level analysis
        # Calculate sector performance
        sector_perf = filtered_df.groupby('sector')['variation'].mean().reset_index()
        top_sector = sector_perf.loc[sector_perf['variation'].idxmax()]
        bottom_sector = sector_perf.loc[sector_perf['variation'].idxmin()]
        
        # Top performing companies
        company_perf = filtered_df.groupby('company')['variation'].mean().reset_index()
        top_company = company_perf.loc[company_perf['variation'].idxmax()]
        
        # Calculate overall market trend
        market_avg = filtered_df.groupby('date')['last_price'].mean().reset_index()
        start_avg = market_avg.iloc[0]['last_price']
        end_avg = market_avg.iloc[-1]['last_price']
        market_change_pct = ((end_avg - start_avg) / start_avg) * 100
        
        # Format the insight with HTML
        insight = [
            html.H4("BVMT Market Overview"),
            html.P([
                f"During the period from ",
                html.Strong(f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, "),
                f"the overall market showed a {'positive' if market_change_pct > 0 else 'negative'} trend with an average price change of ",
                html.Strong(f"{market_change_pct:.2f}%.")
            ]),
            html.P([
                f"The best-performing sector was ",
                html.Strong(f"{top_sector['sector']} with an average change of {top_sector['variation']:.2f}%, "),
                f"while the weakest-performing sector was ",
                html.Strong(f"{bottom_sector['sector']} with an average change of {bottom_sector['variation']:.2f}%.")
            ]),
            html.P([
                f"The top-performing company was ",
                html.Strong(f"{top_company['company']} with an average price change of {top_company['variation']:.2f}%.")
            ]),
            html.H5("Market Interpretation"),
            html.P([
                f"The market has demonstrated {'strength and resilience' if market_change_pct > 0 else 'weakness and volatility'} ",
                f"during this period. Sectors like {top_sector['sector']} have shown particular {'strength' if top_sector['variation'] > 0 else 'stability'}, ",
                f"suggesting {'potential opportunities in these areas.' if top_sector['variation'] > 0 else 'potential for recovery.'}"
            ]),
            html.P([
                "Investors should consider the sector trends when making portfolio decisions, ",
                f"with a focus on {'growth areas' if market_change_pct > 0 else 'defensive positions'} in the current market environment."
            ])
        ]
        
        return html.Div(insight)


@app.callback(
    Output('data-structure-info', 'children'),
    Input('company-dropdown', 'options')
)
def update_data_structure_info(dropdown_options):
    """Generate information about the dataset structure and statistics"""
    
    # Basic dataframe info
    total_rows = len(df)
    total_columns = len(df.columns)
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # In MB
    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
    
    # Count records by company and sector
    company_counts = df['company'].value_counts()
    sector_counts = df['sector'].value_counts()
    
    # Data types
    dtypes = df.dtypes.astype(str)
    
    # Basic statistics for numerical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    stats = df[numeric_cols].describe().round(2)
    
    # Null values count
    null_counts = df.isnull().sum()
    
    # Create the report components
    info_card = dbc.Card([
        dbc.CardBody([
            html.H5("Dataset Overview"),
            dbc.Row([
                dbc.Col([
                    html.P(f"Total Rows: {total_rows}"),
                    html.P(f"Total Columns: {total_columns}"),
                    html.P(f"Memory Usage: {memory_usage:.2f} MB"),
                    html.P(f"Date Range: {date_range}")
                ], width=6),
                dbc.Col([
                    html.P(f"Number of Companies: {len(company_counts)}"),
                    html.P(f"Number of Sectors: {len(sector_counts)}"),
                    html.P(f"Most Active Company: {company_counts.index[0]} ({company_counts.iloc[0]} records)"),
                    html.P(f"Largest Sector: {sector_counts.index[0]} ({sector_counts.iloc[0]} records)")
                ], width=6)
            ])
        ])
    ], className="mb-3")
    
    # Data types card
    dtypes_card = dbc.Card([
        dbc.CardHeader("Column Data Types"),
        dbc.CardBody([
            html.Div([
                html.P(f"{col}: {dtype}") for col, dtype in zip(dtypes.index, dtypes.values)
            ])
        ])
    ], className="mb-3")
    
    # Column descriptions and metadata
    column_descriptions = {
        'date': 'The trading date',
        'company': 'The company symbol/ticker',
        'sector': 'Industry sector of the company',
        'last_price': 'Closing price of the stock for the trading day (in TND)',
        'opening_price': 'Opening price of the stock for the trading day (in TND)',
        'high': 'Highest price of the stock during the trading day (in TND)',
        'low': 'Lowest price of the stock during the trading day (in TND)',
        'volume': 'Number of shares traded during the day',
        'variation': 'Percentage change in price from previous trading day'
    }
    
    # Column information and metadata card
    metadata_card = dbc.Card([
        dbc.CardHeader("Column Descriptions and Metadata"),
        dbc.CardBody([
            dbc.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Column"),
                        html.Th("Type"),
                        html.Th("% Non-Null"),
                        html.Th("Unique Values"),
                        html.Th("Description")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(col),
                        html.Td(dtypes[col]),
                        html.Td(f"{100 - (null_counts[col] / total_rows * 100):.2f}%"),
                        html.Td(str(df[col].nunique()) if col in df.columns else "N/A"),
                        html.Td(column_descriptions.get(col, "No description available"))
                    ]) for col in dtypes.index
                ])
            ], bordered=True, hover=True, responsive=True, striped=True)
        ])
    ], className="mb-3")
    
    # Null values card
    null_card = dbc.Card([
        dbc.CardHeader("Null Values Count"),
        dbc.CardBody([
            html.Div([
                html.P(f"{col}: {count}") for col, count in zip(null_counts.index, null_counts.values) if count > 0
            ] or [html.P("No null values found in the dataset")])
        ])
    ], className="mb-3")
    
    # Statistics card
    stats_card = dbc.Card([
        dbc.CardHeader("Numerical Statistics"),
        dbc.CardBody([
            dbc.Table.from_dataframe(stats, striped=True, bordered=True, hover=True, size="sm")
        ])
    ], className="mb-3")
    
    # Correlation matrix
    corr_card = dbc.Card([
        dbc.CardHeader("Correlation Matrix"),
        dbc.CardBody([
            dcc.Graph(
                figure=px.imshow(
                    df[numeric_cols].corr(),
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    title="Correlation Matrix of Numerical Variables"
                ).update_layout(height=600)
            )
        ])
    ], className="mb-3")
    
    # Numerical columns distributions
    dist_card = dbc.Card([
        dbc.CardHeader("Numerical Columns Distributions"),
        dbc.CardBody([
            dcc.Graph(
                figure=make_distribution_plots(df, numeric_cols)
            )
        ])
    ], className="mb-3")
    
    # Data sample card
    sample_card = dbc.Card([
        dbc.CardHeader("Data Sample (First 5 Rows)"),
        dbc.CardBody([
            dbc.Table.from_dataframe(df.head(5), striped=True, bordered=True, hover=True, size="sm")
        ])
    ])
    
    return html.Div([info_card, metadata_card, dtypes_card, null_card, stats_card, corr_card, dist_card, sample_card])


def make_distribution_plots(dataframe, columns):
    """Create distribution plots for numerical columns"""
    # Limit to 6 columns max to avoid overcrowding
    if len(columns) > 6:
        columns = columns[:6]
    
    # Calculate number of rows and columns for subplots
    n_cols = min(2, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=columns)
    
    # Add histograms for each column
    for i, col in enumerate(columns):
        row = i // n_cols + 1
        col_pos = i % n_cols + 1
        
        # Create histogram trace
        fig.add_trace(
            go.Histogram(x=dataframe[col], name=col),
            row=row, col=col_pos
        )
        
        # Update layout for this subplot
        fig.update_xaxes(title_text=col, row=row, col=col_pos)
        fig.update_yaxes(title_text="Count", row=row, col=col_pos)
    
    # Update overall layout
    fig.update_layout(
        height=300 * n_rows,
        title_text="Distribution of Numerical Variables",
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


@app.callback(
    Output('company-dropdown', 'options'),
    [Input('sector-dropdown', 'value')]
)
def update_company_options(selected_sector):
    """Update company dropdown options based on selected sector"""
    if selected_sector:
        # Filter companies by selected sector
        sector_companies = sorted(df[df['sector'] == selected_sector]['company'].unique())
        return [{'label': company, 'value': company} for company in sector_companies]
    else:
        # If no sector selected, show all companies
        return [{'label': company, 'value': company} for company in sorted(df['company'].unique())]


def main():
    parser = argparse.ArgumentParser(description='Run the BVMT Stock Analysis Dashboard')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port to run the server on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host address to bind to')
    
    args = parser.parse_args()
    
    print(f"Starting BVMT Stock Analysis Dashboard...")
    print(f"Loading data from: {DATA_FILE}")
    print(f"Processed data path: {PROCESSED_DATA_FILE}")
    print(f"Dashboard will be available at: http://{args.host}:{args.port}/")
    
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == '__main__':
    main() 