import pandas as pd
from sklearn.datasets import fetch_openml

from preprocessing import clean_df
from pattern import PatternValidator
from presence import PresenceValidator
from permanence import PermanenceValidator
from logic import LogicValidator
from trust_update import update_trust

def run_demo(X, y, name, best_params):
    # Initialize validators
    pattern    = PatternValidator()
    presence   = PresenceValidator()
    permanence = PermanenceValidator()
    logic      = LogicValidator()

    # Run PPP loop
    history = update_trust(
        X, y,
        pattern, presence, permanence, logic,
        iterations=10,
        alpha=best_params["alpha"],
        beta= best_params["beta"],
        gamma=best_params["gamma"],
        delta=best_params["delta"]
    )

    acc   = history["accuracy"][-1]
    trust = history["T"][-1]
    print(f"{name}: Accuracy={acc:.3f}, Trust={trust:.3f}")

if __name__ == "__main__":
    # 1. Heart Disease demo
    df = pd.read_csv("data/heart_disease_dataset_new.csv")
    df = clean_df(df, target_col="target")
    heart_params = {
        "alpha": 0.19899969870482181,
        "beta":  0.3023754795178367,
        "gamma": 0.15131886131549113,
        "delta": 0.24770213456675536
    }
    run_demo(
        df.drop(columns=["target"]).values,
        df["target"].values,
        "Heart Dataset",
        heart_params
    )

    # 2. Pima Diabetes demo
    df2 = pd.read_csv("data/diabetes.csv")
    df2 = clean_df(df2, target_col="Outcome")
    run_demo(
        df2.drop(columns=["Outcome"]).values,
        df2["Outcome"].values,
        "Pima Diabetes",
        heart_params
    )

    # 3. MNIST-784 demo (no preprocessing)
    mn = fetch_openml("mnist_784", version=1)
    run_demo(
        mn.data.values,
        mn.target.astype(int).values,
        "MNIST-784",
        heart_params
    )

    # 4. UCI Heart Failure demo
    df3 = pd.read_csv("data/heart_failure.csv")
    df3 = clean_df(df3, target_col="DEATH_EVENT")
    run_demo(
        df3.drop(columns=["DEATH_EVENT"]).values,
        df3["DEATH_EVENT"].values,
        "UCI Heart Failure",
        heart_params
    )

