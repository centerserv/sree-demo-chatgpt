import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from pattern import PatternValidator
from presence import PresenceValidator
from permanence import PermanenceValidator
from logic import LogicValidator
from trust_update import update_trust

def run_demo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pattern = PatternValidator(); presence = PresenceValidator()
    permanence = PermanenceValidator(); logic = LogicValidator()

    # tune and get best params (alpha/beta/gamma/delta)
    best = update_trust(X_test, y_test, pattern, presence, permanence, logic)
    print("Best params:", best)

    # then re-run PPP loop with those params and print final metrics (omitted for brevity)

if __name__=="__main__":
    # Example on Heart CSV
    df = pd.read_csv("data/heart_disease_dataset_new.csv")
    run_demo(df.drop(columns=["target"]).values, df["target"].values)

    # Example on MNIST-784
    mn = fetch_openml("mnist_784", version=1)
    run_demo(mn.data.values, mn.target.astype(int).values)

