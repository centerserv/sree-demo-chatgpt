import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from pattern import PatternValidator
from presence import PresenceValidator
from permanence import PermanenceValidator
from logic import LogicValidator
from trust_update import update_trust

def run_demo(X, y, name, best_params):
    # split once
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # init validators
    pattern   = PatternValidator()
    presence  = PresenceValidator()
    permanence= PermanenceValidator()
    logic     = LogicValidator()

    # run PPP loop with your tuned params
    history = update_trust(
        X_test, y_test,
        pattern, presence, permanence, logic,
        iterations=10,
        alpha=best_params["alpha"],
        beta= best_params["beta"],
        gamma=best_params["gamma"],
        delta=best_params["delta"]
    )

    acc = history["accuracy"][-1]
    trust = history["T"][-1]
    print(f"{name}: Accuracy={acc:.3f}, Trust={trust:.3f}")

if __name__ == "__main__":
    # Example on Heart CSV (replace with your own dataset path)
    df = pd.read_csv("data/heart_disease_dataset_new.csv")
    heart_params = {
        "alpha": 0.19899969870482181,
        "beta":  0.3023754795178367,
        "gamma": 0.15131886131549113,
        "delta": 0.24770213456675536
    }
    run_demo(df.drop(columns=["target"]).values, df["target"].values, "Heart Dataset", heart_params)

    # Example on MNIST-784
    mn = fetch_openml("mnist_784", version=1)
    mnist_params = {
        "alpha": 0.19899969870482181,
        "beta":  0.3023754795178367,
        "gamma": 0.15131886131549113,
        "delta": 0.24770213456675536
    }
    run_demo(mn.data.values, mn.target.astype(int).values, "MNIST-784", mnist_params)

