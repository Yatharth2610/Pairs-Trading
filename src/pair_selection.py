"""
Pair selection module using Clustering and Cointegration.
"""
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from statsmodels.tsa.stattools import coint
from tqdm import tqdm

def find_clusters(features, n_clusters=4, method='agglomerative'):
    """
    Cluster assets based on return features.
    """
    X_feat = StandardScaler().fit_transform(features)
    
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
        
    features_copy = features.copy()
    features_copy["cluster"] = model.fit_predict(X_feat)
    return features_copy

def select_cointegrated_pairs(log_prices, clustered_features, p_value_threshold=0.05):
    """
    Find cointegrated pairs only among assets in the same cluster.
    """
    candidate_pairs = []
    for c in clustered_features.cluster.unique():
        names = clustered_features[clustered_features.cluster == c].index.tolist()
        candidate_pairs += list(combinations(names, 2))
        
    print(f"Total candidate pairs from clusters: {len(candidate_pairs)}")
    
    valid_pairs = []
    # Using tqdm for progress bar
    for s1, s2 in tqdm(candidate_pairs, desc="Testing Cointegration"):
        if s1 in log_prices.columns and s2 in log_prices.columns:
            try:
                # Engle-Granger Cointegration Test
                _, pval, _ = coint(log_prices[s1], log_prices[s2])
                if pval < p_value_threshold:
                    valid_pairs.append((s1, s2))
            except Exception as e:
                pass
                
    print(f"Total valid cointegrated pairs: {len(valid_pairs)}")
    return valid_pairs
