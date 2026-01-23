"""
Data loading and preprocessing for S&P 500 portfolio optimization.

This module handles loading the S&P 500 monthly dataset and preparing it
for portfolio optimization workflows.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings


def load_sp500_data(filepath: str = "data/sp500_monthly.csv") -> pd.DataFrame:
    """
    Load the S&P 500 monthly dataset.
    
    Expected columns:
    - Returns, prices, shares outstanding
    - SIC codes, tickers
    - Membership start/end dates
    - Date index
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with cleaned S&P 500 data
    """
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime if present
    date_cols = ['date', 'Date', 'DATE', 'datadate', 'time']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break
    
    # Ensure we have a date index
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to infer from other columns
        if 'year' in df.columns and 'month' in df.columns:
            df.index = pd.to_datetime(df[['year', 'month']].assign(day=1))
        else:
            warnings.warn("Could not identify date column. Please check data format.")
    
    return df


def prepare_returns_data(
    df: pd.DataFrame,
    return_col: str = 'ret',
    ticker_col: str = 'ticker',
    date_index: bool = True
) -> pd.DataFrame:
    """
    Extract and prepare returns data in wide format (assets x dates).
    
    Args:
        df: Raw dataframe with returns
        return_col: Name of return column
        ticker_col: Name of ticker/identifier column
        date_index: Whether index is dates
        
    Returns:
        Wide-format DataFrame with assets as columns, dates as index
    """
    if return_col not in df.columns:
        # Try common alternatives
        alt_cols = ['return', 'RET', 'retx', 'RETX', 'retadj', 'RETADJ']
        for col in alt_cols:
            if col in df.columns:
                return_col = col
                break
        else:
            raise ValueError(f"Could not find return column. Available: {df.columns.tolist()}")
    
    if ticker_col not in df.columns:
        alt_cols = ['TICKER', 'permno', 'PERMNO', 'gvkey', 'GVKEY']
        for col in alt_cols:
            if col in df.columns:
                ticker_col = col
                break
        else:
            raise ValueError(f"Could not find ticker column. Available: {df.columns.tolist()}")
    
    # Pivot to wide format: dates x assets
    returns_wide = df.pivot_table(
        values=return_col,
        index=df.index if date_index else None,
        columns=ticker_col,
        aggfunc='first'  # Handle duplicates if any
    )
    
    # Sort by date
    if date_index:
        returns_wide = returns_wide.sort_index()
    
    return returns_wide


def filter_valid_assets(
    returns: pd.DataFrame,
    min_periods: int = 24,  # Minimum 2 years of data
    max_missing_pct: float = 0.2  # Max 20% missing
) -> pd.DataFrame:
    """
    Filter assets to keep only those with sufficient data.
    
    Args:
        returns: Wide-format returns DataFrame (dates x assets)
        min_periods: Minimum number of non-null observations
        max_missing_pct: Maximum percentage of missing values
        
    Returns:
        Filtered returns DataFrame
    """
    # Calculate missing percentage
    missing_pct = returns.isnull().sum() / len(returns)
    
    # Filter assets
    valid_assets = (
        (returns.count() >= min_periods) & 
        (missing_pct <= max_missing_pct)
    )
    
    filtered = returns.loc[:, valid_assets]
    
    print(f"Filtered from {len(returns.columns)} to {len(filtered.columns)} assets")
    print(f"Removed {len(returns.columns) - len(filtered.columns)} assets with insufficient data")
    
    return filtered


def create_asset_universe(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_membership_dates: bool = True
) -> pd.DataFrame:
    """
    Create asset universe based on S&P 500 membership dates.
    
    Args:
        df: Raw dataframe
        start_date: Start date for universe (YYYY-MM-DD)
        end_date: End date for universe (YYYY-MM-DD)
        use_membership_dates: Whether to filter by membership start/end dates
        
    Returns:
        Filtered dataframe with only assets in universe at each date
    """
    filtered = df.copy()
    
    # Filter by date range
    if start_date:
        filtered = filtered[filtered.index >= start_date]
    if end_date:
        filtered = filtered[filtered.index <= end_date]
    
    # Filter by membership dates if available
    if use_membership_dates:
        membership_start_cols = ['start_date', 'START_DATE', 'begdate', 'BEGDATE']
        membership_end_cols = ['end_date', 'END_DATE', 'enddate', 'ENDDATE']
        
        start_col = None
        end_col = None
        
        for col in membership_start_cols:
            if col in filtered.columns:
                start_col = col
                break
        
        for col in membership_end_cols:
            if col in filtered.columns:
                end_col = col
                break
        
        if start_col and end_col:
            # Filter: asset must be in S&P 500 at this date
            mask = (
                (filtered[start_col].isna() | (filtered.index >= pd.to_datetime(filtered[start_col]))) &
                (filtered[end_col].isna() | (filtered.index <= pd.to_datetime(filtered[end_col])))
            )
            filtered = filtered[mask]
    
    return filtered


def get_data_summary(returns: pd.DataFrame) -> dict:
    """
    Get summary statistics for the returns data.
    
    Args:
        returns: Wide-format returns DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'n_assets': len(returns.columns),
        'n_periods': len(returns),
        'date_range': (returns.index.min(), returns.index.max()),
        'missing_pct': returns.isnull().sum().sum() / (len(returns) * len(returns.columns)),
        'mean_return': returns.mean().mean(),
        'std_return': returns.std().mean(),
        'min_return': returns.min().min(),
        'max_return': returns.max().max(),
    }
    
    return summary


def print_data_summary(returns: pd.DataFrame):
    """Print a formatted summary of the returns data."""
    summary = get_data_summary(returns)
    
    print("=" * 60)
    print("S&P 500 Dataset Summary")
    print("=" * 60)
    print(f"Number of assets: {summary['n_assets']}")
    print(f"Number of periods: {summary['n_periods']}")
    print(f"Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"Missing data: {summary['missing_pct']:.2%}")
    print(f"Mean return: {summary['mean_return']:.4f}")
    print(f"Std return: {summary['std_return']:.4f}")
    print(f"Min return: {summary['min_return']:.4f}")
    print(f"Max return: {summary['max_return']:.4f}")
    print("=" * 60)
