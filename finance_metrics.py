"""
Compute per-symbol market features and ensure labeled dataset exists.
Reads:  data/processed/merged_features.csv  (must contain date, symbol, adj_close)
Writes: data/processed/labeled_features.csv (adds/keeps target_label and market features)
"""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH  = os.path.join(BASE_DIR, "data", "processed", "merged_features.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "labeled_features.csv")

def require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"merged_features.csv missing required columns: {missing}")

def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-symbol returns, moving averages, and rolling volatility using vectorized transforms.
    (No groupby.apply; compatible with older pandas.)
    """
    df = df.copy()
    df = df.sort_values(["symbol", "date"], kind="mergesort")

    df["ret_1d"] = (
        df.groupby("symbol")["adj_close"]
          .pct_change(1, fill_method=None)
    )
    df["ret_5d"] = (
        df.groupby("symbol")["adj_close"]
          .pct_change(5, fill_method=None)
    )

    df["ma_5d"] = (
        df.groupby("symbol")["adj_close"]
          .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )
    df["ma_20d"] = (
        df.groupby("symbol")["adj_close"]
          .transform(lambda s: s.rolling(20, min_periods=1).mean())
    )

    daily_ret = (
        df.groupby("symbol")["adj_close"]
          .pct_change(fill_method=None)
    )
    df["vol_10d"] = (
        daily_ret.groupby(df["symbol"])
                 .transform(lambda s: s.rolling(10, min_periods=2).std())
    )

    return df

def ensure_target_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a string label column 'target_label' exists.
    If already present, keep it. Otherwise derive from next-day return: BUY if fwd>0 else SELL.
    """
    if "target_label" in df.columns:
        return df

    df = df.copy().sort_values(["symbol", "date"], kind="mergesort")
    fwd_px = df.groupby("symbol")["adj_close"].shift(-1)
    fwd_ret = (fwd_px / df["adj_close"]) - 1.0

    # Use pandas to avoid dtype promotion issues with strings + NaN
    target_label = pd.Series(index=df.index, dtype="object")
    target_label[fwd_ret > 0]  = "BUY"
    target_label[fwd_ret <= 0] = "SELL"

    df["target_label"] = target_label
    return df

def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Merged feature file not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    require_cols(df, ["date", "symbol", "adj_close"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Compute market features (vectorized)
    df = add_market_features(df)

    # Keep NLP features as-is; ensure label exists
    df = ensure_target_label(df)

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Labeled features with market metrics saved to {OUT_PATH}")

if __name__ == "__main__":
    main()