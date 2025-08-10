# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def _is_binary(series: pd.Series) -> bool:
    """Return True if series contains only two unique values {0,1} (int or float)."""
    if pd.api.types.is_bool_dtype(series):
        return True
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    uniq = set(np.unique(s.values))
    return uniq.issubset({0, 1, 0.0, 1.0}) and 1 < len(uniq) <= 2

def clean_df(df: pd.DataFrame, target_col: str, use_smote: bool = True) -> pd.DataFrame:
    """
    Preprocess any binary-target CSV (8+ features):
    1) Coerce feature columns to numeric (where possible).
    2) Median-impute zeros/NaNs on non-binary features.
    3) Clip outliers to 1–99 percentile on non-binary features.
    4) Z-score normalize (float32).
    5) Optional SMOTE if class imbalance >60/40 and imblearn is available.
    """
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    feature_cols = [c for c in df.columns if c != target_col]

    # Coerce non-binary features to numeric (strings -> NaN)
    for col in feature_cols:
        if not _is_binary(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1) Median-impute zeros/NaNs on non-binary features
    for col in feature_cols:
        if _is_binary(df[col]):
            continue
        med = df.loc[(df[col] != 0) & df[col].notna(), col].median()
        if pd.isna(med):
            med = df[col].median()
        df[col] = df[col].replace(0, med).fillna(med)

    # 2) Clip outliers (1–99 percentile) on non-binary features
    for col in feature_cols:
        if _is_binary(df[col]):
            continue
        low, high = df[col].quantile([0.01, 0.99])
        if pd.isna(low) or pd.isna(high) or low == high:
            continue
        df[col] = df[col].clip(low, high)

    # 3) Z-score normalize (float32)
    feats = df[feature_cols].astype(np.float32, copy=False)
    scaler = StandardScaler().fit(feats)
    df[feature_cols] = scaler.transform(feats).astype(np.float32)

    # Ensure target is numeric 0/1
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    # 4) Optional SMOTE
    y = df[target_col]
    imbalance = abs(y.mean() - 0.5) > 0.1
    if use_smote and imbalance:
        try:
            from imblearn.over_sampling import SMOTE  # lazy import
            sm = SMOTE(random_state=42)
            Xb, yb = sm.fit_resample(df[feature_cols], y)
            out = pd.DataFrame(Xb, columns=feature_cols)
            out[target_col] = yb.astype(int)
            return out
        except Exception as e:
            print(f"[warn] SMOTE unavailable; continuing without oversampling. Reason: {e}")

    return df
