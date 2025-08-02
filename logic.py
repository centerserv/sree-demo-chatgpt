import numpy as np
from sklearn.preprocessing import StandardScaler

class LogicValidator:
    def validate(self, preds, X, dataset_name=None):
        X_norm = StandardScaler().fit_transform(X)
        classes = np.unique(preds)
        means = {c: X_norm[preds==c].mean(axis=0) for c in classes}
        V_l = []
        sims = []
        for pred, x in zip(preds, X_norm):
            mean_vec = means[pred]
            cos = np.dot(x, mean_vec) / (np.linalg.norm(x)*np.linalg.norm(mean_vec)+1e-10)
            sims.append(cos)
        μ, σ = np.mean(sims), np.std(sims)
        for cos in sims:
            V_l.append(1.0 if (μ-3*σ) <= cos <= (μ+3*σ) else 0.0)
        return np.array(V_l)

