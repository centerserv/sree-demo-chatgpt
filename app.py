# app.py
import sys, os, glob
import numpy as np
import pandas as pd
import streamlit as st

# Import local modules
sys.path.append(".")
from preprocessing import clean_df
from pattern import PatternValidator
from presence import PresenceValidator
from permanence import PermanenceValidator
from logic import LogicValidator
from trust_update import update_trust

def is_binary_col(s: pd.Series) -> bool:
    """Return True if a column has only two unique (0/1) values (after coercion)."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return False
    vals = set(np.unique(s.values))
    return vals.issubset({0, 1, 0.0, 1.0}) and 1 < len(vals) <= 2

def load_fallback_csv() -> str:
    """Return path to a fallback CSV present in the repo or return None."""
    candidates = [
        "data/heart_failure.csv",
        "UCI_heart_failure_clinical_records_dataset(2).csv",
        "heart_disease_dataset.csv",
        "Cardiovascular_Disease_Dataset.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    any_csvs = glob.glob("*.csv") + glob.glob("data/*.csv")
    return any_csvs[0] if any_csvs else None

def main():
    st.set_page_config(page_title="SREE PPP Demo", layout="wide")
    st.title("SREE Phase 1 Demo — Pattern · Presence · Permanence")

    # Sidebar controls (always define before use)
    with st.sidebar:
        st.header("Run settings")
        uploaded_file = st.file_uploader("Upload CSV (binary target, 8+ features)", type=["csv"])
        # We'll set the target column later after reading the CSV and detecting candidates
        # The rest of the widgets will appear after the data is loaded

    # Load dataset
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.caption("Using uploaded file.")
    else:
        fallback_path = load_fallback_csv()
        if fallback_path is None:
            st.error("No CSV found. Upload a file in the sidebar or add one to the repo (e.g., data/heart_failure.csv).")
            return
        df = pd.read_csv(fallback_path)
        st.caption(f"Using fallback dataset: {fallback_path}")

    # Normalize column names (trim spaces)
    df.columns = [str(c).strip() for c in df.columns]
    st.write("### Data preview")
    st.dataframe(df.head())
    st.write("Columns:", list(df.columns))

    # Determine binary target candidates
    binary_candidates = [col for col in df.columns if is_binary_col(df[col])]
    # Common label names to prioritize
    common_names = ["DEATH_EVENT", "target", "Outcome", "Class", "label", "y"]
    ordered = [c for c in common_names if c in df.columns] + [c for c in binary_candidates if c not in common_names]
    if not ordered:
        st.error("No binary target column detected. Please upload a CSV with a 0/1 label column.")
        return

    # Show remaining sidebar options now that we have data
    with st.sidebar:
        target_col = st.selectbox("Target column", options=ordered, index=0)
        iterations = st.slider("PPP iterations", 5, 100, 30, 5)
        tune_trials = st.slider("Auto-tune trials", 0, 30, 10, 1, help="0 = skip tuning")
        use_smote  = st.checkbox("Use SMOTE when imbalance >60/40", value=True)
        st.caption("Defaults if tuning is 0:")
        alpha = st.number_input("alpha", 0.0, 1.0, 0.20, 0.01)
        beta  = st.number_input("beta",  0.0, 1.0, 0.42, 0.01)
        gamma = st.number_input("gamma", 0.0, 1.0, 0.15, 0.01)
        delta = st.number_input("delta", 0.0, 1.0, 0.20, 0.01)
        run_button = st.button("Run PPP")

    if not run_button:
        return

    # Clean the dataset
    try:
        df_clean = clean_df(df.copy(), target_col=target_col, use_smote=use_smote)
    except TypeError:
        # fallback for older clean_df signature
        df_clean = clean_df(df.copy(), target_col=target_col)

    if target_col not in df_clean.columns:
        st.error(f"Target column '{target_col}' not found after cleaning.")
        return

    X = df_clean.drop(columns=[target_col]).values
    y = df_clean[target_col].values

    # Prepare validators
    pattern    = PatternValidator()
    presence   = PresenceValidator()      # entropy-based V_q
    permanence = PermanenceValidator()    # hashed ledger V_b
    logic      = LogicValidator()

    # Auto-tune if requested
    if tune_trials > 0:
        st.info(f"Auto-tuning ({tune_trials} trials)…")
        try:
            best = update_trust(
                X, y, pattern, presence, permanence, logic,
                iterations=10,
                alpha=None, beta=None, gamma=None, delta=None,
                n_trials=tune_trials
            )
        except TypeError:
            best = update_trust(
                X, y, pattern, presence, permanence, logic,
                iterations=10,
                alpha=None, beta=None, gamma=None, delta=None
            )
        alpha, beta, gamma, delta = best["alpha"], best["beta"], best["gamma"], best["delta"]
        st.success(f"Best params: {best}")

    # Run the final PPP loop
    st.info(f"Running PPP loop for {iterations} iterations…")
    history = update_trust(
        X, y,
        pattern, presence, permanence, logic,
        iterations=iterations,
        alpha=alpha, beta=beta, gamma=gamma, delta=delta
    )

    # Prepare and display metrics
    acc_series = pd.Series(history["accuracy"], name="Accuracy")
    t_series   = pd.Series(history["T"],         name="Trust")
    metrics_df = pd.concat([acc_series, t_series], axis=1)
    st.write("### Per-iteration metrics")
    st.line_chart(metrics_df)
    st.write(metrics_df)

    st.success(f"Final: Accuracy={metrics_df['Accuracy'].iloc[-1]:.3f}, Trust={metrics_df['Trust'].iloc[-1]:.3f}")

    # Offer CSV download
    st.download_button(
        "Download metrics CSV",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="sree_ppp_metrics.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
