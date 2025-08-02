import hashlib
import numpy as np

class PermanenceValidator:
    def __init__(self):
        self.ledger = {}

    def validate(self, preds, probs):
        V_b = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            prob_str = ",".join(f"{p:.6f}" for p in prob)
            h = hashlib.sha256(f"{pred}|{prob_str}".encode()).hexdigest()
            V_b.append(1.0 if (i not in self.ledger or self.ledger[i] == h) else 0.0)
            self.ledger[i] = h
        return np.array(V_b)

