# logic.py
import numpy as np
from sklearn.preprocessing import StandardScaler

class LogicValidator:
    """
    Computes V_l by measuring cosine similarity of each sample to its class mean,
    z-score normalizing features. Flags 1 if within μ ± 3σ, else 0.
    """

    def validate(self, preds, X):
        if preds.ndim != 1:
            preds = preds.ravel()
        X_norm = StandardScaler().fit_transform(X)
        classes = np.unique(preds)
        means = {c: X_norm[preds == c].mean(axis=0) for c in classes}
        sims, V_l = [], []
        for pred, x in zip(preds, X_norm):
            mean_vec = means[pred]
            cos = np.dot(x, mean_vec) / (
                np.linalg.norm(x) * np.linalg.norm(mean_vec) + 1e-8
            )
            sims.append(cos)
        μ, σ = np.mean(sims), np.std(sims)
        for cos in sims:
            V_l.append(1.0 if (μ - 3 * σ) <= cos <= (μ + 3 * σ) else 0.0)
        return np.array(V_l)
