"""
Backtesting and performance evaluation module.
Strictly formatted according to PEP 8 standards.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def identify_volatility_regime(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Identify high/low volatility regimes.
    
    Args:
        returns: DataFrame of daily log returns for the assets.
        window: Rolling window size for volatility calculation.
        
    Returns:
        Series of integers: 1 for High Volatility, 0 for Low Volatility.
    """
    rolling_vol = returns.rolling(window).std().mean(axis=1)
    median_vol = rolling_vol.median()
    regime = (rolling_vol > median_vol).astype(int)
    return regime

def compute_position_sizes(ml_prob: np.ndarray, vol_regime: pd.Series, valid_indices: list) -> np.ndarray:
    """
    Compute position sizes scaled by ML confidence, penalizing high volatility regimes.
    
    Args:
        ml_prob: Array of calibrated ML probabilities.
        vol_regime: Series of volatility regimes (1=High, 0=Low).
        valid_indices: List of indices aligning ML predictions with the original time series.
        
    Returns:
        Array of normalized position sizes [0, 1].
    """
    aligned_regimes = vol_regime.iloc[valid_indices].values
    
    # Scale position size down in high volatility regimes
    confidence = ml_prob * (1 - aligned_regimes[-len(ml_prob):])
    
    max_conf = np.nanmax(confidence)
    if max_conf > 0:
        position_size = confidence / max_conf
    else:
        position_size = np.zeros_like(confidence)
        
    return position_size

def calculate_pair_pnl(spread: np.ndarray, positions: np.ndarray, position_sizes: np.ndarray) -> np.ndarray:
    """
    Calculate Mark-to-Market continuous daily PnL for a single pair over the out-of-sample period.
    This naturally avoids a step-graph because PnL is accrued daily while the position is held.
    
    Args:
        spread: Array of historical spread values.
        positions: Array of target positions (-1, 0, 1).
        position_sizes: Array of dynamic position sizing multipliers.
        
    Returns:
        Array of daily scaled PnL.
    """
    # Shift positions by 1 to represent executing at the close and realizing return the next day
    daily_pnl = positions[:-1] * np.diff(spread)
    
    # Align to the out-of-sample test period
    test_pnl = daily_pnl[-len(position_sizes):]
    
    # Scale by the dynamic ML position size
    scaled_pnl = test_pnl * position_sizes
    
    return scaled_pnl

def calculate_trade_metrics(positions: np.ndarray, scaled_pnl: np.ndarray) -> tuple:
    """
    Extract discrete trades from the continuous position array to compute trade-level metrics.
    
    Args:
        positions: Array of positions over time.
        scaled_pnl: Array of daily PnL.
        
    Returns:
        Tuple containing (total_trades, win_rate).
    """
    # Align positions array length with scaled_pnl
    aligned_positions = positions[-(len(scaled_pnl) + 1):]
    
    trades = []
    current_trade_pnl = 0.0
    in_trade = False
    
    for i in range(1, len(aligned_positions)):
        prev_pos = aligned_positions[i - 1]
        curr_pos = aligned_positions[i]
        
        # Accrue PnL if we held a position going into today
        if prev_pos != 0:
            current_trade_pnl += scaled_pnl[i - 1]
            
        # Trade initiation or continuation
        if prev_pos == 0 and curr_pos != 0:
            in_trade = True
            
        # Trade exit or reversal
        if in_trade and (curr_pos == 0 or curr_pos != prev_pos):
            if prev_pos != 0:  # Valid trade closure
                trades.append(current_trade_pnl)
                current_trade_pnl = 0.0
                in_trade = False
                
            # If reversal, a new trade immediately begins
            if curr_pos != 0:
                in_trade = True

    # Handle a trade left open at the end of the backtest
    if in_trade and current_trade_pnl != 0:
        trades.append(current_trade_pnl)

    total_trades = len(trades)
    if total_trades > 0:
        winning_trades = sum(1 for t in trades if t > 0)
        win_rate = (winning_trades / total_trades) * 100.0
    else:
        win_rate = 0.0
        
    return total_trades, win_rate

def calculate_metrics(portfolio_pnl: np.ndarray, positions: np.ndarray) -> dict:
    """
    Calculate backtest performance metrics.
    Assumes an initial capital of 1.0 (100%) to calculate accurate percentage drawdowns.
    
    Args:
        portfolio_pnl: Array of daily portfolio PnL.
        positions: Array of historical positions to calculate discrete trade metrics.
        
    Returns:
        Dictionary containing all performance metrics.
    """
    returns_series = pd.Series(portfolio_pnl)
    
    # 1. Sharpe Ratio
    daily_std = returns_series.std()
    sharpe = (np.sqrt(252) * returns_series.mean() / daily_std) if daily_std != 0 else 0.0
    
    # 2. Continuous Equity Curve (Starting at 1.0 representing 100% capital)
    equity_curve = 1.0 + returns_series.cumsum()
    total_pnl_pct = (equity_curve.iloc[-1] - 1.0) * 100.0
    
    # 3. Maximum Drawdown (Strictly as a percentage of peak equity)
    running_max = equity_curve.cummax()
    drawdown_pct = (running_max - equity_curve) / running_max
    max_dd_pct = drawdown_pct.max() * 100.0
    
    # 4. Trade-level metrics
    total_trades, win_rate = calculate_trade_metrics(positions, portfolio_pnl)
    
    metrics = {
        "Total PnL (%)": total_pnl_pct,
        "Total Trades": total_trades,
        "Max Drawdown (%)": max_dd_pct,
        "Sharpe Ratio": sharpe,
        "Win Rate (%)": win_rate,
        "equity_curve": equity_curve,
        "drawdown_series": drawdown_pct * 100.0 # Store as percentage
    }
    
    return metrics

def print_performance_dashboard(metrics: dict):
    """
    Print a cleanly formatted dashboard of the performance metrics.
    """
    print("="*40)
    print("      PERFORMANCE METRICS DASHBOARD     ")
    print("="*40)
    print(f"Total PnL         : {metrics['Total PnL (%)']:.2f}%")
    print(f"Total Trades      : {metrics['Total Trades']}")
    print(f"Max Drawdown      : {metrics['Max Drawdown (%)']:.2f}%")
    print(f"Sharpe Ratio      : {metrics['Sharpe Ratio']:.2f}")
    print(f"Win Rate          : {metrics['Win Rate (%)']:.2f}%")
    print("="*40)

def plot_performance(metrics: dict):
    """
    Plot continuous Mark-to-Market equity curve and percentage drawdown.
    """
    equity_curve = metrics['equity_curve']
    drawdown_series = metrics['drawdown_series']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Equity Curve
    equity_curve.plot(ax=axes[0], color='blue', linewidth=1.5)
    axes[0].set_title("Mark-to-Market Equity Curve (Continuous)", fontsize=12)
    axes[0].set_ylabel("Equity Multiplier (1.0 = Base)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Drawdown
    drawdown_series.plot(ax=axes[1], color='red', linewidth=1.5)
    axes[1].fill_between(drawdown_series.index, drawdown_series, 0, color='red', alpha=0.3)
    axes[1].set_title("Drawdown (%)", fontsize=12)
    axes[1].set_ylabel("Drawdown %")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].invert_yaxis() # Invert y-axis to visually represent drops
    
    plt.tight_layout()
    plt.show()
