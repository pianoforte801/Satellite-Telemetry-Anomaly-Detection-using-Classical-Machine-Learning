# src/preprocessing.py
# Preprocessing aligned with your original notebook logic

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ------------------------------------------------
# Build Anomaly_Bin (same logic as notebook)
# ------------------------------------------------
def add_anomaly_bin(df):
    df = df.copy()
    anomaly_cols = [c for c in df.columns if c.lower().startswith("anomaly_")]

    if len(anomaly_cols) > 0:
        for c in anomaly_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        df["Anomaly_Bin"] = (df[anomaly_cols].sum(axis=1) > 0).astype(int)

    return df


# ------------------------------------------------
# Fit preprocessing on labeled datasets only
# ------------------------------------------------
def fit_preprocessing(flight_df, tvac_df, pca_variance=0.95):

    # Build labels
    flight_df = add_anomaly_bin(flight_df)
    tvac_df = add_anomaly_bin(tvac_df)

    # Keep numeric features
    Xf = flight_df.select_dtypes(include=[np.number]).drop(columns=["Anomaly_Bin"], errors="ignore")
    Xt = tvac_df.select_dtypes(include=[np.number]).drop(columns=["Anomaly_Bin"], errors="ignore")

    # Remove anomaly_* columns from features
    Xf = Xf.drop(columns=[c for c in Xf.columns if "anomaly" in c.lower()], errors="ignore")
    Xt = Xt.drop(columns=[c for c in Xt.columns if "anomaly" in c.lower()], errors="ignore")

    # Common columns between datasets
    common_cols = sorted(list(set(Xf.columns) & set(Xt.columns)))
    Xf = Xf[common_cols]
    Xt = Xt[common_cols]

    # Combine labeled data
    X_combined = pd.concat([Xf, Xt], axis=0)

    # Fill missing with 0 (same as notebook)
    X_combined = X_combined.fillna(0)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # PCA (retain variance instead of fixed 2)
    pca = PCA(n_components=pca_variance, random_state=42)
    pca.fit(X_scaled)

    return {
        "columns": common_cols,
        "scaler": scaler,
        "pca": pca
    }


# ------------------------------------------------
# Transform any dataset (Flight / TVAC / Orbit)
# ------------------------------------------------
def transform_dataset(df, prep):

    df = add_anomaly_bin(df)

    X = df.select_dtypes(include=[np.number]).drop(columns=["Anomaly_Bin"], errors="ignore")
    X = X.drop(columns=[c for c in X.columns if "anomaly" in c.lower()], errors="ignore")

    # Align columns
    X = X.reindex(columns=prep["columns"])

    # Fill missing with 0
    X = X.fillna(0)

    # Scale + PCA
    X_scaled = prep["scaler"].transform(X)
    X_pca = prep["pca"].transform(X_scaled)

    return X_pca


# ------------------------------------------------
# PCA reconstruction error for orbit anomaly scoring
# ------------------------------------------------
def pca_reconstruction_error(df, prep):

    df = add_anomaly_bin(df)

    X = df.select_dtypes(include=[np.number]).drop(columns=["Anomaly_Bin"], errors="ignore")
    X = X.drop(columns=[c for c in X.columns if "anomaly" in c.lower()], errors="ignore")
    X = X.reindex(columns=prep["columns"])
    X = X.fillna(0)

    X_scaled = prep["scaler"].transform(X)

    Z = prep["pca"].transform(X_scaled)
    X_reconstructed = prep["pca"].inverse_transform(Z)

    error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

    return error