# pattern.py
from sklearn.neural_network import MLPClassifier

class PatternValidator:
    def __init__(self):
        # Increased iterations, fixed seed, LBFGS solver for small data
        self.model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=500,
            solver='lbfgs',
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def validate(self, X, y):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        accuracy = (preds == y).mean()
        return preds, probs, accuracy

