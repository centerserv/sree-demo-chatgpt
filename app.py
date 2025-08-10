# app.py
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Make sure local modules import cleanly
sys.path.append(".")
from preprocessing import clean_df
from pattern import PatternValidator
from presence import PresenceValidator
from permanence import PermanenceValidator
from logic import LogicValidator
from trust_update import update_trust

st.set_page_config(page_title="SREE Demo (PPP + Trust)", layout="wide")

st.title("SREE Phase 1 Demo: Pattern–Presence–Permanence (PPP)")
st.caption("Upload any binary-class CSV (8+ columns). Defaults to UCI Heart Failure if no file provided.")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Run Settings")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    target_col = st.text_input("Target column (0/1)", value="DEATH_EVENT")
    iterations = st.slider("PPP iterations", 5, 100, 30, 5)
    tune_trials = st.slider("Auto-tune trials (Optuna)", 0, 30, 10, 1,
                            help="0 = skip tuning and use defaults below")
    st.markdown("---")
    st.caption("Defaults if tuning is 0")
    alpha = st.number_input("alpha", 0.0, 1.0, 0.20, 0.01)
    beta  = st.number_input("beta",  0.0, 1.0, 0.42, 0.01)
    gamma = st.number_input("gamma", 0.0, 1.0, 0.15, 0.01)
    delta = st.number_input("delta", 0.0, 1.0, 0.20, 0.01)
    run = st.button("Run PPP")

# --- Load data ---
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    # Fallback to sample in repo
    df = pd.read_csv("data/heart_failure.csv")

st.write("### Data Preview", df.head())

# --- Run PPP ---
if run:
    # Clean & split
    df_clean = clean_df(df.copy(), target_col=target_col)
    X = df_clean.drop(columns=[target_col]).values
    y = df_clean[target_col].values

    # Validators
    pattern    = PatternValidator()
    presence   = PresenceValidator()      # entropy-based V_q
    permanence = PermanenceValidator()    # your current permanence (can upgrade later)
    logic      = LogicValidator()

    # Tuning or fixed params
    if tune_trials and tune_trials > 0:
        st.info(f"Auto-tuning ({tune_trials} trials)…")
        best = update_trust(
            X, y,
            pattern, presence, permanence, logic,
            iterations=10,        # short inner loop while tuning
            alpha=None, beta=None, gamma=None, delta=None,
            n_trials=tune_trials  # your trust_update should accept this param; if not, it will ignore
        )
        st.success(f"Best params: {best}")
        alpha, beta, gamma, delta = best["alpha"], best["beta"], best["gamma"], best["delta"]

    # Final run
    st.info(f"Running PPP loop for {iterations} iterations…")
    history = update_trust(
        X, y,
        pattern, presence, permanence, logic,
        iterations=iterations,
        alpha=alpha, beta=beta, gamma=gamma, delta=delta
    )

    acc_series = pd.Series(history["accuracy"], name="Accuracy")
    t_series   = pd.Series(history["T"],         name="Trust")
    out = pd.concat([acc_series, t_series], axis=1)
    st.write("### Per-iteration metrics", out)

    # Charts
    st.line_chart(out)

    # Final numbers
    st.success(f"Final: Accuracy={out['Accuracy'].iloc[-1]:.3f}, Trust={out['Trust'].iloc[-1]:.3f}")

    # Download
    st.download_button(
        "Download metrics CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="sree_ppp_metrics.csv",
        mime="text/csv"
    )

