# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def _is_binary(series: pd.Series) -> bool:
    """Treat bools, ints, and floats that are only {0,1} (allowing 0.0/1.0) as binary."""
    if pd.api.types.is_bool_dtype(series):
        return True
    # coerce to numeric (non-numeric -> NaN), then drop NaN and check unique set
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    uniq = set(np.unique(s.values))
    return uniq.issubset({0, 1, 0.0, 1.0}) and len(uniq) <= 2

def clean_df(df: pd.DataFrame, target_col: str, use_smote: bool = True) -> pd.DataFrame:
    """
    Generic cleaning for any binary-target CSV (8+ features):
      1) Coerce non-target columns to numeric (errors->NaN)
      2) Median-impute zeros/NaNs on non-binary features
      3) Clip outliers to 1–99 percentile on non-binary features
      4) Z-score normalize features (float32 to keep memory light)
      5) Optional SMOTE if class imbalance > 60/40 (lazy import, safe fallback)
    """
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # 0) Coerce non-target columns to numeric where possible (strings -> NaN)
    feature_cols = [c for c in df.columns if c != target_col]
    for col in feature_cols:
        # Skip coercion for clearly-binary columns; otherwise make numeric
        if not _is_binary(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1) Median-impute zeros/NaNs on non-binary features
    for col in feature_cols:
        if _is_binary(df[col]):
            continue
        med = df.loc[df[col] != 0, col].median()
        if pd.isna(med):
            med = df[col].median()
        df[col] = df[col].replace(0, med).fillna(med)

    # 2) Clip outliers (1–99 percentile) on non-binary features
    for col in feature_cols:
        if _is_binary(df[col]):
            continue
        low, high = df[col].quantile([0.01, 0.99])
        # handle degenerate columns
        if pd.isna(low) or pd.isna(high) or low == high:
            continue
        df[col] = df[col].clip(low, high)

    # 3) Z-score normalize features (cast to float32 to reduce memory/CPU)
    feats = df[feature_cols].astype(np.float32, copy=False)
    scaler = StandardScaler().fit(feats)
    df[feature_cols] = scaler.transform(feats).astype(np.float32)

    # Ensure target is 0/1 numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    # 4) Optional SMOTE if imbalance > 60/40 and imblearn is available
    y = df[target_col]
    imbalance = abs(y.mean() - 0.5) > 0.1
    if use_smote and imbalance:
        try:
            from imblearn.over_sampling import SMOTE  # lazy import (won't crash if unavailable)
            sm = SMOTE(random_state=42)
            Xb, yb = sm.fit_resample(df[feature_cols], y)
            out = pd.DataFrame(Xb, columns=feature_cols)
            out[target_col] = yb.astype(int)
            return out
        except Exception as e:
            # Safe fallback: continue without oversampling
            print(f"[warn] SMOTE unavailable or incompatible; continuing without oversampling. Reason: {e}")

    return df
