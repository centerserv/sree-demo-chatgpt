# main.py
import pandas as pd

from preprocessing import clean_df
from pattern import PatternValidator
from presence import PresenceValidator
from permanence import PermanenceValidator
from logic import LogicValidator
from trust_update import update_trust

if __name__ == "__main__":
    # 1. Load and preprocess UCI Heart Failure data
    df = pd.read_csv("data/heart_failure.csv")
    df = clean_df(df, target_col="DEATH_EVENT")
    X = df.drop(columns=["DEATH_EVENT"]).values
    y = df["DEATH_EVENT"].values

    # 2. Auto-tune hyperparameters (20 iterations)
    print("Tuning hyperparameters on UCI Heart Failure...")
    pattern    = PatternValidator()
    presence   = PresenceValidator()
    permanence = PermanenceValidator()
    logic      = LogicValidator()
    tuned_params = update_trust(
        X, y,
        pattern, presence, permanence, logic,
        iterations=20,
        alpha=None, beta=None, gamma=None, delta=None
    )
    print("Best params:", tuned_params)

    # 3. Run PPP loop with tuned params and more iterations (50)
    print("Running PPP loop with tuned params (50 iterations)...")
    history = update_trust(
        X, y,
        pattern, presence, permanence, logic,
        iterations=50,
        alpha=tuned_params["alpha"],
        beta= tuned_params["beta"],
        gamma=tuned_params["gamma"],
        delta=tuned_params["delta"]
    )

    # 4. Print per-iteration self-refinement
    print("\nSelf-refinement per iteration:")
    for i, (acc, t) in enumerate(zip(history["accuracy"], history["T"]), start=1):
        print(f"Iteration {i:02d}: Accuracy={acc:.3f}, Trust={t:.3f}")

    # 5. Print final metrics
    final_acc   = history["accuracy"][-1]
    final_trust = history["T"][-1]
    print(f"\nFinal UCI Heart Failure: Accuracy={final_acc:.3f}, Trust={final_trust:.3f}")

