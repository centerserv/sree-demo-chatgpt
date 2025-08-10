# app.py
import sys, os, glob
import numpy as np
import pandas as pd
import streamlit as st

# local modules
sys.path.append(".")
from preprocessing import clean_df
from pattern import PatternValidator
from presence import PresenceValidator
from permanence import PermanenceValidator
from logic import LogicValidator
from trust_update import update_trust

st.set_page_config(page_title="SREE Demo (PPP + Trust)", layout="wide")
st.title("SREE Phase 1 Demo — Pattern · Presence · Permanence")

# ---- Sidebar (create widgets first!) ----
with st.sidebar:
    st.header("Run settings")
    uploaded   = st.file_uploader("Upload CSV (binary target, 8+ features)", type=["csv"])
    target_col = st.text_input("Target column", value="DEATH_EVENT")
    iterations = st.slider("PPP iterations", 5, 100, 30, 5)
    tune_trials = st.slider("Auto-tune trials", 0, 30, 10, 1, help="0 = skip tuning")
    use_smote  = st.checkbox("Use SMOTE when imbalanced (>60/40)", value=True)
    st.caption("Defaults if tuning is 0:")
    alpha = st.number_input("alpha", 0.0, 1.0, 0.20, 0.01)
    beta  = st.number_input("beta",  0.0, 1.0, 0.42, 0.01)
    gamma = st.number_input("gamma", 0.0, 1.0, 0.15, 0.01)
    delta = st.number_input("delta", 0.0, 1.0, 0.20, 0.01)
    run = st.button("Run PPP")

# ---- Load data (upload OR fallback in repo) ----
def load_fallback_csv():
    candidates = [
        "data/heart_failure.csv",
        "UCI_heart_failure_clinical_records_dataset(2).csv",
        "heart_disease_dataset.csv",
        "Cardiovascular_Disease_Dataset.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    any_csvs = glob.glob("*.csv") + glob.glob("data/*.csv")
    return any_csvs[0] if any_csvs else None

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.caption("Using uploaded file.")
else:
    path = load_fallback_csv()
    if not path:
        st.error("No CSV found. Upload a file in the sidebar or add a CSV to the repo (e.g., data/heart_failure.csv).")
        st.stop()
    df = pd.read_csv(path)
    st.caption(f"Using fallback dataset: {path}")

st.write("### Data preview")
st.dataframe(df.head())

# ---- Run PPP ----
if run:
    # Clean
    try:
        df_clean = clean_df(df.copy(), target_col=target_col, use_smote=use_smote)
    except TypeError:
        # for older preprocessing.py without use_smote arg
        df_clean = clean_df(df.copy(), target_col=target_col)

    # Guard: target column exists and is binary-ish
    if target_col not in df_clean.columns:
        st.error(f"Target column '{target_col}' not found.")
        st.stop()

    X = df_clean.drop(columns=[target_col]).values
    y = df_clean[target_col].values

    # Validators
    pattern    = PatternValidator()
    presence   = PresenceValidator()      # entropy-based
    permanence = PermanenceValidator()    # current permanence
    logic      = LogicValidator()

    # Optional tuning
    if tune_trials > 0:
        st.info(f"Auto-tuning ({tune_trials} trials)…")
        try:
            best = update_trust(
                X, y, pattern, presence, permanence, logic,
                iterations=10, alpha=None, beta=None, gamma=None, delta=None,
                n_trials=tune_trials
            )
        except TypeError:
            best = update_trust(
                X, y, pattern, presence, permanence, logic,
                iterations=10, alpha=None, beta=None, gamma=None, delta=None
            )
        st.success(f"Best params: {best}")
        alpha, beta, gamma, delta = best["alpha"], best["beta"], best["gamma"], best["delta"]

    # Final run
    st.info(f"Running PPP loop for {iterations} iterations…")
    history = update_trust(
        X, y, pattern, presence, permanence, logic,
        iterations=iterations, alpha=alpha, beta=beta, gamma=gamma, delta=delta
    )

    acc_series = pd.Series(history["accuracy"], name="Accuracy")
    t_series   = pd.Series(history["T"],         name="Trust")
    out = pd.concat([acc_series, t_series], axis=1)

    st.write("### Per-iteration metrics")
    st.line_chart(out)
    st.write(out)

    st.success(f"Final: Accuracy={out['Accuracy'].iloc[-1]:.3f}, Trust={out['Trust'].iloc[-1]:.3f}")

    st.download_button(
        "Download metrics CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="sree_ppp_metrics.csv",
        mime="text/csv"
    )
