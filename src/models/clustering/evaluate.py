"""
evaluate.py — Clustering evaluation metrics for RFM segmentation.

Provides ``evaluate_clustering(X, labels)`` which computes two standard
internal clustering validity indices:

- **Silhouette Score** (higher is better, range −1 to +1):
  Measures how similar each point is to its own cluster compared with other
  clusters. Values closer to +1 indicate well-separated, cohesive clusters.

- **Davies-Bouldin Index** (lower is better, minimum 0):
  Average ratio of within-cluster scatter to between-cluster separation.
  Lower values indicate better-defined clusters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score


def evaluate_clustering(
    X: "np.ndarray | pd.DataFrame",
    labels: "np.ndarray | list[int]",
) -> dict[str, float]:
    """Compute Silhouette Score and Davies-Bouldin Index for a clustering result.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix used for clustering. Can be a NumPy array or a
        pandas DataFrame.
    labels : array-like of shape (n_samples,)
        Cluster labels assigned to each sample (e.g. from KMeans.fit_predict
        or DBSCAN.fit_predict). Noise points (DBSCAN label −1) are silently
        excluded from both metrics.

    Returns
    -------
    dict with keys:
        ``silhouette``     : float — Silhouette Score  (higher is better)
        ``davies_bouldin`` : float — Davies-Bouldin Index (lower is better)

    Raises
    ------
    ValueError
        If the number of unique non-noise labels is less than 2 (metrics
        require at least 2 clusters).
    """
    X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
    labels_arr = np.asarray(labels)

    # Exclude DBSCAN noise points (label == -1)
    mask = labels_arr != -1
    X_clean = X_arr[mask]
    labels_clean = labels_arr[mask]

    n_clusters = len(set(labels_clean))
    if n_clusters < 2:
        raise ValueError(
            f"evaluate_clustering requires at least 2 clusters, got {n_clusters}. "
            "Check that the clustering produced meaningful results."
        )

    sil = float(silhouette_score(X_clean, labels_clean))
    db = float(davies_bouldin_score(X_clean, labels_clean))

    return {"silhouette": sil, "davies_bouldin": db}
