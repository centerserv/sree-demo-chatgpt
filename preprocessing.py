import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def clean_df(df, target_col):
    # 1. Median‐impute zeros and NaNs on non-binary columns
    for col in df.columns:
        if col == target_col:
            continue
        unique_vals = set(df[col].dropna().unique())
        if unique_vals <= {0, 1}:
            continue
        med = df.loc[(df[col] != 0) & df[col].notna(), col].median()
        df[col] = df[col].replace(0, med).fillna(med)

    # 2. Clip outliers to the 1–99 percentile range on non-binary columns
    for col in df.columns:
        if col == target_col:
            continue
        unique_vals = set(df[col].dropna().unique())
        if unique_vals <= {0, 1}:
            continue
        low, high = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(low, high)

    # 3. Z‐score normalize features (cast to float first)
    features = df.drop(columns=[target_col]).astype(float)
    scaler  = StandardScaler().fit(features)
    df[features.columns] = scaler.transform(features)

    # 4. Balance classes if imbalance >60/40
    y = df[target_col]
    if abs(y.mean() - 0.5) > 0.1:
        sm = SMOTE(random_state=42)
        X_bal, y_bal = sm.fit_resample(df.drop(columns=[target_col]), y)
        df = pd.DataFrame(X_bal, columns=features.columns)
        df[target_col] = y_bal

    return df

