# --- Load data ---
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    import os, glob
    candidates = [
        "data/heart_failure.csv",
        "UCI_heart_failure_clinical_records_dataset(2).csv",
        "heart_disease_dataset.csv",
        "Cardiovascular_Disease_Dataset.csv",
    ]
    found = next((p for p in candidates if os.path.exists(p)), None)
    if not found:
        # any CSV in repo root or data/
        any_csvs = glob.glob("*.csv") + glob.glob("data/*.csv")
        found = any_csvs[0] if any_csvs else None
    if not found:
        st.error("No CSV found. Upload a file in the sidebar or add one to the repo (e.g., data/heart_failure.csv).")
        st.stop()
    df = pd.read_csv(found)
    st.caption(f"Using fallback dataset: {found}")
