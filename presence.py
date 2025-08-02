import numpy as np

class PresenceValidator:
    def validate(self, probs):
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        d = probs.shape[1]
        V_q = 1 - entropy / np.log(d)
        return V_q

