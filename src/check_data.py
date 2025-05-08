#!/usr/bin/env python
import pandas as pd

def main():
    """Check the data in the CSV file"""
    file_path = 'data/bvmt_stocks_2025.csv'

    # Load the data
    df = pd.read_csv(file_path)

    # Print summary
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Companies: {len(df['company'].unique())}")
    print(f"Dates: {len(df['date'].unique())}")
    print(f"Expected records: {len(df['company'].unique()) * len(df['date'].unique())}")

    # Check columns
    print(f"\nColumns: {df.columns.tolist()}")

    # Check data types
    print("\nData types:")
    print(df.dtypes)

    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Check statistics for numeric columns
    print("\nStatistics for numeric columns:")
    print(df.describe())

    # Check statistics by sector
    print("\nAverage price by sector:")
    sector_avg_price = df.groupby('sector')['last_price'].mean().sort_values(ascending=False)
    print(sector_avg_price)

    # Check if we have data for all companies for each day
    companies = df['company'].unique()
    dates = df['date'].unique()

    print(f"\nChecking if we have data for all {len(companies)} companies for each of the {len(dates)} days...")

    # Count records for each company
    company_counts = df['company'].value_counts()
    print("\nCompany record counts (first 10):")
    print(company_counts.head(10))

    # Check if any company has fewer records than expected
    companies_with_missing_data = company_counts[company_counts < len(dates)]
    if not companies_with_missing_data.empty:
        print(f"\nCompanies with missing data: {len(companies_with_missing_data)}")
        print(companies_with_missing_data)
    else:
        print("\nAll companies have data for all dates!")

    # Count records for each date
    date_counts = df['date'].value_counts()
    print("\nDate record counts (first 10):")
    print(date_counts.head(10))

    # Check if any date has fewer records than expected
    dates_with_missing_data = date_counts[date_counts < len(companies)]
    if not dates_with_missing_data.empty:
        print(f"\nDates with missing data: {len(dates_with_missing_data)}")
        print(dates_with_missing_data)
    else:
        print("\nAll dates have data for all companies!")

if __name__ == "__main__":
    main()
