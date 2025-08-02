from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class PatternValidator:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200)

    def fit(self, X, y):
        self.model.fit(X, y)

    def validate(self, X, y):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        accuracy = (preds == y).mean()
        return preds, probs, accuracy

