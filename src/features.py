"""
Feature extraction for portfolio optimization.

This module extracts predictive features from returns and price data,
including momentum, volatility, drawdown, and other technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings


def compute_momentum(
    returns: pd.DataFrame,
    lookback: int = 12,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Compute momentum as cumulative return over lookback period.
    
    Args:
        returns: Wide-format returns DataFrame (dates x assets)
        lookback: Number of periods to look back
        normalize: Whether to normalize by volatility
        
    Returns:
        Momentum features (dates x assets)
    """
    # Cumulative return over lookback period
    momentum = (1 + returns).rolling(window=lookback).apply(np.prod, raw=True) - 1
    
    if normalize:
        # Normalize by volatility (Sharpe-like)
        vol = returns.rolling(window=lookback).std() * np.sqrt(12)  # Annualized
        momentum = momentum / (vol + 1e-8)  # Add small epsilon to avoid division by zero
    
    return momentum


def compute_volatility(
    returns: pd.DataFrame,
    lookback: int = 12,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Compute rolling volatility.
    
    Args:
        returns: Wide-format returns DataFrame
        lookback: Number of periods to look back
        annualize: Whether to annualize (multiply by sqrt(12) for monthly)
        
    Returns:
        Volatility features (dates x assets)
    """
    vol = returns.rolling(window=lookback).std()
    
    if annualize:
        vol = vol * np.sqrt(12)  # Annualize monthly returns
    
    return vol


def compute_drawdown(
    returns: pd.DataFrame,
    lookback: int = 12
) -> pd.DataFrame:
    """
    Compute maximum drawdown over lookback period.
    
    Args:
        returns: Wide-format returns DataFrame
        lookback: Number of periods to look back
        
    Returns:
        Maximum drawdown features (dates x assets)
    """
    # Cumulative returns
    cumret = (1 + returns).cumprod()
    
    # Rolling maximum
    rolling_max = cumret.rolling(window=lookback, min_periods=1).max()
    
    # Drawdown
    drawdown = (cumret - rolling_max) / rolling_max
    
    # Maximum drawdown over period
    max_drawdown = drawdown.rolling(window=lookback).min()
    
    return max_drawdown


def compute_mean_reversion(
    returns: pd.DataFrame,
    lookback: int = 12
) -> pd.DataFrame:
    """
    Compute mean reversion signal (negative autocorrelation).
    
    Args:
        returns: Wide-format returns DataFrame
        lookback: Number of periods to look back
        
    Returns:
        Mean reversion features (dates x assets)
    """
    # Recent return vs. longer-term average
    recent_ret = returns.rolling(window=3).mean()  # Last 3 months
    long_ret = returns.rolling(window=lookback).mean()  # Last 12 months
    
    # Mean reversion: negative if recent > long-term (overvalued)
    mean_rev = -(recent_ret - long_ret)
    
    return mean_rev


def compute_liquidity_proxy(
    returns: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None,
    shares: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute liquidity proxy (if price/volume data available).
    
    For now, uses return volatility as a proxy for liquidity.
    More sophisticated versions could use bid-ask spreads, volume, etc.
    
    Args:
        returns: Wide-format returns DataFrame
        prices: Optional price data
        shares: Optional shares outstanding data
        
    Returns:
        Liquidity proxy features (dates x assets)
    """
    # Simple proxy: inverse of volatility (higher vol = lower liquidity)
    vol = compute_volatility(returns, lookback=3, annualize=False)
    liquidity = 1 / (vol + 1e-8)
    
    # Normalize
    liquidity = (liquidity - liquidity.mean()) / (liquidity.std() + 1e-8)
    
    return liquidity


def extract_all_features(
    returns: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None,
    shares: Optional[pd.DataFrame] = None,
    feature_config: Optional[Dict] = None
) -> Dict[str, pd.DataFrame]:
    """
    Extract all features for portfolio optimization.
    
    Args:
        returns: Wide-format returns DataFrame (dates x assets)
        prices: Optional price data
        shares: Optional shares outstanding data
        feature_config: Optional configuration dict with feature parameters
        
    Returns:
        Dictionary of feature DataFrames, each with shape (dates x assets)
    """
    if feature_config is None:
        feature_config = {
            'momentum_lookback': 12,
            'volatility_lookback': 12,
            'drawdown_lookback': 12,
            'mean_reversion_lookback': 12,
        }
    
    features = {}
    
    # Momentum
    features['momentum'] = compute_momentum(
        returns,
        lookback=feature_config.get('momentum_lookback', 12)
    )
    
    # Volatility
    features['volatility'] = compute_volatility(
        returns,
        lookback=feature_config.get('volatility_lookback', 12)
    )
    
    # Drawdown
    features['drawdown'] = compute_drawdown(
        returns,
        lookback=feature_config.get('drawdown_lookback', 12)
    )
    
    # Mean reversion
    features['mean_reversion'] = compute_mean_reversion(
        returns,
        lookback=feature_config.get('mean_reversion_lookback', 12)
    )
    
    # Liquidity proxy
    features['liquidity'] = compute_liquidity_proxy(returns, prices, shares)
    
    return features


def stack_features(features: Dict[str, pd.DataFrame]) -> np.ndarray:
    """
    Stack features into a 3D array: (time, assets, features).
    
    Args:
        features: Dictionary of feature DataFrames
        
    Returns:
        Numpy array of shape (T, N, F) where:
        - T = number of time periods
        - N = number of assets
        - F = number of features
    """
    # Get common index and columns (dates and assets)
    feature_names = list(features.keys())
    first_feat = features[feature_names[0]]
    
    # Stack features
    stacked = np.stack([
        features[name].values for name in feature_names
    ], axis=-1)  # Shape: (T, N, F)
    
    return stacked


def prepare_feature_tensor(
    returns: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None,
    shares: Optional[pd.DataFrame] = None,
    feature_config: Optional[Dict] = None
) -> Tuple[np.ndarray, pd.DatetimeIndex, pd.Index]:
    """
    Prepare feature tensor for PyTorch model.
    
    Args:
        returns: Wide-format returns DataFrame
        prices: Optional price data
        shares: Optional shares outstanding data
        feature_config: Optional feature configuration
        
    Returns:
        Tuple of (feature_array, date_index, asset_index)
        - feature_array: (T, N, F) numpy array
        - date_index: DatetimeIndex
        - asset_index: Index of asset names
    """
    # Extract features
    features = extract_all_features(returns, prices, shares, feature_config)
    
    # Stack into tensor
    feature_tensor = stack_features(features)
    
    # Get indices
    first_feat = list(features.values())[0]
    date_index = first_feat.index
    asset_index = first_feat.columns
    
    return feature_tensor, date_index, asset_index
