# main.py
import pandas as pd

from preprocessing import clean_df
from pattern import PatternValidator
from presence import PresenceValidator
from permanence import PermanenceValidator
from logic import LogicValidator
from trust_update import update_trust

def run_demo(X, y, name, best_params, iterations=20):
    # Initialize validators
    pattern    = PatternValidator()
    presence   = PresenceValidator()
    permanence = PermanenceValidator()
    logic      = LogicValidator()

    # Run PPP loop
    history = update_trust(
        X, y,
        pattern, presence, permanence, logic,
        iterations=iterations,
        alpha=best_params["alpha"],
        beta= best_params["beta"],
        gamma=best_params["gamma"],
        delta=best_params["delta"]
    )

    acc   = history["accuracy"][-1]
    trust = history["T"][-1]
    print(f"{name}: Accuracy={acc:.3f}, Trust={trust:.3f}")

if __name__ == "__main__":
    # 1. Load and preprocess UCI Heart Failure data
    df = pd.read_csv("data/heart_failure.csv")
    df = clean_df(df, target_col="DEATH_EVENT")

    # 2. Tuned hyperparameters
    best_params = {
        "alpha": 0.19899969870482181,
        "beta":  0.3023754795178367,
        "gamma": 0.15131886131549113,
        "delta": 0.24770213456675536
    }

    # 3. Run only the UCI Heart Failure demo
    run_demo(
        df.drop(columns=["DEATH_EVENT"]).values,
        df["DEATH_EVENT"].values,
        "UCI Heart Failure",
        best_params
    )

