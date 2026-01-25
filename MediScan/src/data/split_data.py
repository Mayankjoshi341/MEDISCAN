import pandas as pd
def features_split(main_df: pd.DataFrame, target_col="disease"):
    feature_cols = [c for c in main_df.columns if c.startswith("symptom_")]

    if not feature_cols:
        raise ValueError("No symptom columns found")

    if target_col not in main_df.columns:
        raise ValueError(f"{target_col} not found")

    X = main_df[feature_cols]
    y = main_df[target_col]
    print("features split is complete successfully.")

    return X, y