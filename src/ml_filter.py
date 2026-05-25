"""
Machine Learning module for trade quality prediction.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd

def extract_ml_features_and_labels(spread, z, look_ahead=5, look_back_vol=20, look_back_mom=5):
    """
    Extract features and labels for the ML model.
    Label: 1 if the spread reverts towards mean within look_ahead days, 0 otherwise.
    """
    X, y = [], []
    valid_indices = []
    
    # Start after enough history for rolling features
    start_idx = max(look_back_vol, look_back_mom)
    
    for t in range(start_idx, len(z) - look_ahead):
        if np.isnan(z[t]):
            continue
            
        # Features
        feature_vec = [
            z[t],
            abs(z[t]),
            np.std(spread[t-look_back_vol:t]),
            np.mean(np.diff(spread[t-look_back_mom:t]))
        ]
        
        # Label: successful mean reversion
        # If z is positive, we expect spread to go down. So spread[t+look_ahead] < spread[t] -> negative change.
        # spread_change * np.sign(z[t]) < 0 indicates reversion.
        spread_change = spread[t+look_ahead] - spread[t]
        label = int(spread_change * np.sign(z[t]) < 0)
        
        X.append(feature_vec)
        y.append(label)
        valid_indices.append(t)
        
    return np.array(X), np.array(y), valid_indices

def train_and_predict_rf(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest classifier with Isotonic calibration.
    Returns the calibrated probabilities and the fitted model.
    """
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    # Calibrate probabilities
    clf = CalibratedClassifierCV(rf, method="isotonic", cv=3)
    clf.fit(X_train, y_train)
    
    ml_prob = clf.predict_proba(X_test)[:, 1]
    
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, ml_prob)
        print(f"Test ROC-AUC: {auc:.4f}")
    else:
        print("Not enough classes in test set for ROC-AUC.")
        
    return ml_prob, clf

def walk_forward_validation(X, y, train_size_pct=0.7):
    """
    Simple walk-forward split (time series split).
    For a more advanced version, a rolling window approach would be used.
    """
    split = int(train_size_pct * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test
