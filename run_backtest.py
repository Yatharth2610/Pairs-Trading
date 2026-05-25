import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import data_loader, pair_selection, spread_model, ml_filter, dl_model, backtest, explainability

def run_pipeline():
    print("--- STARTING PAIRS TRADING PIPELINE ---")
    
    TICKERS = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
        'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'LT.NS', 'BAJFINANCE.NS',
        'HUL.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'M&M.NS', 'TATAMOTORS.NS'
    ]

    print("1. Loading Data...")
    prices = data_loader.download_data(TICKERS, start_date='2018-01-01')
    log_prices, returns = data_loader.preprocess_data(prices)
    features = data_loader.calculate_features(returns)

    print("2. Selecting Pairs...")
    clustered_features = pair_selection.find_clusters(features, n_clusters=3, method='agglomerative')
    valid_pairs = pair_selection.select_cointegrated_pairs(log_prices, clustered_features, p_value_threshold=0.05)
    print(f"Selected Pairs: {valid_pairs}")

    if not valid_pairs:
        print("No valid cointegrated pairs found.")
        return

    target_pair = valid_pairs[0]
    print(f"3. Processing Pair: {target_pair}...")
    
    spread, z, beta = spread_model.compute_spread_and_zscore(log_prices, target_pair)
    positions = spread_model.generate_positions(z)

    print("4. Training Machine Learning Filter...")
    X, y, valid_indices = ml_filter.extract_ml_features_and_labels(spread, z)
    X_train, X_test, y_train, y_test = ml_filter.walk_forward_validation(X, y)
    ml_prob, clf = ml_filter.train_and_predict_rf(X_train, y_train, X_test, y_test)

    print("5. Running Backtest...")
    vol_regime = backtest.identify_volatility_regime(returns)
    position_size = backtest.compute_position_sizes(ml_prob, vol_regime, valid_indices)
    
    ml_test_start_idx = valid_indices[len(X_train)]
    pnl = backtest.calculate_pair_pnl(spread, positions, position_size)
    
    metrics = backtest.calculate_metrics(pnl, positions)
    
    backtest.print_performance_dashboard(metrics)
    print("Pipeline execution complete. (Charts omitted in headless run)")

if __name__ == "__main__":
    run_pipeline()
