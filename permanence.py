# permanence.py
import hashlib
import numpy as np

class PermanenceValidator:
    """
    Simulates a Merkle-proof ledger: stores SHA-256 hashes of (pred, prob)
    tuples keyed by row index. V_b=1 on first insertion or if hash unchanged;
    V_b=0 if the hash mismatches prior ledger entry.
    """

    def __init__(self):
        self.ledger = {}

    def validate(self, preds, probs):
        V_b = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            prob_str = ",".join(f"{p:.6f}" for p in prob)
            h = hashlib.sha256(f"{pred}|{prob_str}".encode()).hexdigest()
            if i not in self.ledger or self.ledger[i] == h:
                V_b.append(1.0)
            else:
                V_b.append(0.0)
            self.ledger[i] = h
        return np.array(V_b)
