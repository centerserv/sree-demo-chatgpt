# presence.py
import numpy as np
from scipy.stats import entropy

class PresenceValidator:
    """
    Computes V_q from normalized Shannon entropy of class probabilities.
    V_q = 1 - H(p) / log2(C), per sample, in [0,1].
    Higher V_q = higher confidence (lower entropy).
    """

    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon

    def score(self, probs):
        """
        Args:
            probs: ndarray shape (n_samples, n_classes) of predicted probabilities
        Returns:
            vq_per_sample: ndarray shape (n_samples,)
            mean_vq: float
        """
        if probs.ndim != 2:
            probs = np.atleast_2d(probs)

        # Clip and renormalize for numerical stability
        p = np.clip(probs, self.epsilon, 1.0)
        p = p / p.sum(axis=1, keepdims=True)

        # Per-sample entropy (base-2), then normalize by max entropy log2(C)
        ent = entropy(p, base=2, axis=1)
        max_ent = np.log2(p.shape[1]) if p.shape[1] > 1 else 1.0
        vq = 1.0 - (ent / max_ent)

        return vq, float(vq.mean())

    # Keep a validate() alias for compatibility with the trust loop
    def validate(self, probs):
        return self.score(probs)
