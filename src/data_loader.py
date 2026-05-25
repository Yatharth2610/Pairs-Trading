"""
Data loading and preprocessing module.
"""
import yfinance as yf
import pandas as pd
import numpy as np

def download_data(tickers, start_date="2018-01-01", end_date=None):
    """
    Download Adjusted Close prices for a list of tickers.
    """
    print(f"Downloading data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)["Adj Close"]
    data = data.dropna(axis=1, how='all') # Drop columns that are completely NA
    # Forward fill then backward fill to handle missing days
    data = data.ffill().bfill()
    return data

def preprocess_data(prices):
    """
    Calculate log prices and daily returns.
    """
    log_prices = np.log(prices)
    returns = log_prices.diff().dropna()
    return log_prices, returns

def calculate_features(returns):
    """
    Calculate rolling features for clustering (volatility, mean return, skewness, kurtosis).
    """
    features = pd.DataFrame({
        "vol": returns.std(),
        "mean_ret": returns.mean(),
        "skew": returns.skew(),
        "kurt": returns.kurt()
    })
    return features.dropna()
