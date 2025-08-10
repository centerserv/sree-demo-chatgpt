# --- Load data ---
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.caption("Using uploaded file.")
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
        any_csvs = glob.glob("*.csv") + glob.glob("data/*.csv")
        found = any_csvs[0] if any_csvs else None
    if not found:
        st.error("No CSV found. Please upload a file or add one to the repo.")
        st.stop()
    df = pd.read_csv(found)
    st.caption(f"Using fallback dataset: {found}")

# Normalize column names (trim stray spaces etc.)
df.columns = [str(c).strip() for c in df.columns]

st.write("### Data preview")
st.dataframe(df.head())
st.write("Columns:", list(df.columns))

# --- Target selection (auto-suggest) ---
def _is_binary_col(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: 
        return False
    u = set(np.unique(s.values))
    return u.issubset({0,1,0.0,1.0}) and 1 < len(u) <= 2

# Prioritize common label names if present
common_names = ["DEATH_EVENT", "target", "Outcome", "Class", "label", "y"]
binary_cols  = [c for c in df.columns if _is_binary_col(df[c])]
ordered = [c for c in common_names if c in df.columns] + [c for c in binary_cols if c not in common_names]
options = ordered or list(df.columns)

# Sidebar widget (replaces your previous target input)
target_col = st.sidebar.selectbox("Target column", options=options, index=0)

# Keep the rest of your sidebar widgets (iterations, tuning, SMOTE, Run PPP)
