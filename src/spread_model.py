"""
Dynamic Spread Modeling using Kalman Filter.
"""
import numpy as np
import pandas as pd

def kalman_beta(y, x):
    """
    Estimate dynamic hedge ratio (beta) using a 1D Kalman Filter.
    """
    beta, P = 0.0, 1.0
    Q, R = 1e-5, 0.001
    betas = []

    for i in range(len(y)):
        P += Q
        K = P * x[i] / (x[i]**2 * P + R)
        beta += K * (y[i] - beta * x[i])
        P *= (1 - K * x[i])
        betas.append(beta)

    return np.array(betas)

def compute_spread_and_zscore(log_prices, pair, window=20):
    """
    Compute the spread and rolling z-score for a given pair.
    Uses rolling z-score to avoid look-ahead bias instead of full-sample z-score.
    """
    y = log_prices[pair[0]].values
    x = log_prices[pair[1]].values
    
    # Dynamic beta
    beta = kalman_beta(y, x)
    spread = y - beta * x
    spread_series = pd.Series(spread)
    
    # Rolling Z-score to prevent data leakage
    rolling_mean = spread_series.rolling(window=window).mean()
    rolling_std = spread_series.rolling(window=window).std()
    
    z = (spread_series - rolling_mean) / rolling_std
    
    return spread, z.values, beta

def generate_positions(z, entry_z=2.0, exit_z=0.5):
    """
    Generate target positions based on z-score thresholds.
    1: Long Spread (y underpriced, x overpriced)
    -1: Short Spread (y overpriced, x underpriced)
    0: Flat
    """
    pos = np.zeros(len(z))
    for i in range(1, len(z)):
        if np.isnan(z[i]):
            continue
            
        if z[i] > entry_z:
            pos[i] = -1
        elif z[i] < -entry_z:
            pos[i] = 1
        elif abs(z[i]) < exit_z:
            pos[i] = 0
        else:
            pos[i] = pos[i-1] # Maintain position
    return pos
