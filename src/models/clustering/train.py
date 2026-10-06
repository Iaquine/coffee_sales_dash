"""
train.py — K-Means training for RFM clustering.

Provides ``train_kmeans(df_rfm, k)`` which trains a scikit-learn KMeans model
on the normalised RFM features and returns the fitted model together with the
cluster labels assigned to each row.

The normalised feature columns used for clustering are:
    recency_norm, frequency_norm, monetary_norm
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# Feature columns used for clustering (normalised to [0, 1])
RFM_FEATURE_COLS = ["recency_norm", "frequency_norm", "monetary_norm"]


def train_kmeans(
    df_rfm: pd.DataFrame,
    k: int,
    random_state: int = 42,
    n_init: int = 10,
) -> tuple[KMeans, np.ndarray]:
    """Train a K-Means model on normalised RFM features.

    Parameters
    ----------
    df_rfm : pd.DataFrame
        DataFrame containing at least the columns in ``RFM_FEATURE_COLS``
        (``recency_norm``, ``frequency_norm``, ``monetary_norm``).
        Typically the output of :func:`src.features.rfm_features.compute_rfm`.
    k : int
        Number of clusters.
    random_state : int
        Seed for reproducibility (default 42).
    n_init : int
        Number of K-Means initialisations (default 10).

    Returns
    -------
    model : sklearn.cluster.KMeans
        Fitted KMeans instance. Access centroids via ``model.cluster_centers_``.
    labels : np.ndarray of shape (n_samples,)
        Cluster label (0-indexed) for each row of *df_rfm*.
    """
    X = df_rfm[RFM_FEATURE_COLS].to_numpy()

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(X)
    return model, labels
