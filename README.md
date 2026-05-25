# Statistical + Machine Learning + Deep Learning Driven Pairs Trading System

This project is an advanced, end-to-end quantitative research system for pairs trading, designed for Indian equity markets (NIFTY 50). It combines classical statistical arbitrage techniques with modern Machine Learning and Deep Learning to predict trade quality, size positions dynamically, and maintain explainability.

## Objective
Given a statistically valid pairs trading opportunity, the system predicts whether the spread between two assets will successfully mean-revert within a fixed horizon. The ML layer focuses on **trade quality prediction** rather than raw price prediction.

## Architecture Pipeline
1. **Market Data**: Daily Adjusted Close prices via `yfinance`.
2. **Feature Engineering**: Log prices, returns, rolling volatility, skewness, kurtosis.
3. **Clustering-Based Pair Selection**: Agglomerative/K-Means Clustering to group similar assets before applying Cointegration tests.
4. **Cointegration Testing**: Engle-Granger Cointegration Test (p-value < 0.05).
5. **Kalman Filter Hedge Ratio**: Dynamic hedge ratio estimation to handle structural breaks.
6. **Spread Construction & Z-Score**: Rolling z-score for entry/exit signals.
7. **Machine Learning Trade Filtering**: Random Forest with probability calibration to predict successful mean reversion.
8. **Deep Learning Temporal Modeling**: PyTorch LSTM to model temporal spread dynamics and estimate reversion probability.
9. **Regime-Aware Position Sizing**: Volatility-based regime detection adjusting ML confidence for position sizing.
10. **Walk-Forward Validation**: Strict rolling train-test windows to prevent data leakage.
11. **Explainability**: SHAP (SHapley Additive exPlanations) for model interpretability.

## Project Structure
- `src/`: Modular Python components for the pipeline.
- `notebooks/`: End-to-end Jupyter notebooks for research and demonstration.
- `data/`: Directory for caching downloaded market data.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook in `notebooks/` to see the end-to-end execution.

## Disclaimer
This project is for educational and research purposes only. It is not financial advice.
