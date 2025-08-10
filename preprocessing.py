# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def _is_binary(series: pd.Series) -> bool:
    vals = pd.unique(series.dropna())
    return set(vals).issubset({0, 1}) and len(vals) <= 2

def clean_df(df: pd.DataFrame, target_col: str, use_smote: bool = True) -> pd.DataFrame:
    df = df.copy()

    # 1) Median-impute zeros/NaNs on non-binary feature columns
    for col in df.columns:
        if col == target_col:
            continue
        if _is_binary(df[col]):
            continue
        med = df.loc[df[col] != 0, col].median()
        if pd.isna(med):
            med = df[col].median()
        df[col] = df[col].replace(0, med).fillna(med)

    # 2) Clip outliers (1–99 percentile) on non-binary features
    for col in df.columns:
        if col == target_col or _is_binary(df[col]):
            continue
        low, high = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(low, high)

    # 3) Z-score normalize features
    features = df.drop(columns=[target_col]).astype(float)
    scaler = StandardScaler().fit(features)
    df[features.columns] = scaler.transform(features)

    # 4) Optional SMOTE if imbalance > 60/40 and imblearn is available
    y = df[target_col]
    imbalance = abs(y.mean() - 0.5) > 0.1
    if use_smote and imbalance:
        try:
            from imblearn.over_sampling import SMOTE  # lazy import
            sm = SMOTE(random_state=42)
            Xb, yb = sm.fit_resample(df.drop(columns=[target_col]), y)
            out = pd.DataFrame(Xb, columns=features.columns)
            out[target_col] = yb
            return out
        except Exception as e:
            print(f"[warn] SMOTE unavailable or incompatible; continuing without oversampling. Reason: {e}")

    return df
