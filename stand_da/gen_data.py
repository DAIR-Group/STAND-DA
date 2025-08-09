import numpy as np
from typing import List


def generate_data(mu: float, delta: List[int], n: int, d: int, num_anomalies: int = 5):
    mu = np.full((n, d), mu, dtype=np.float64)
    noise = np.random.normal(loc = 0, scale = 1, size=(n, d))
    Sigma = np.kron(np.eye(d), np.eye(n))
    X = mu + noise
    labels = np.zeros(n)
    if len(delta) == 1:
        n_anomalies = num_anomalies
        idx = np.random.choice(n, n_anomalies, replace=False)
        X[idx] = X[idx] + delta[0]
        if delta[0] != 0:
            labels[idx] = np.ones(n_anomalies)
    else:
        n_anomalies = num_anomalies
        idx = np.random.choice(n, n_anomalies, replace=False)
        if 0 in delta: 
            delta.pop(delta.index(0))
        if len(delta) != 0:
            split_points = sorted(np.random.choice(range(1, len(idx)), len(delta) - 1, replace=False))
            segments = np.split(idx, split_points)
            for i, segment in enumerate(segments):
                X[segment] = X[segment] + delta[i]
            labels[idx] = 1
    return X, labels, Sigma