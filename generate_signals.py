"""
Score latest date per ticker using text + market features + symbol one-hots.
Outputs BUY/SELL/HOLD with probabilities and confidence.
Reads:  data/processed/merged_features.csv
Model:  models/signal_generator/rf_signal_model.pkl
Writes: data/processed/signals_latest.csv
"""

import os
import sys
import pandas as pd
import joblib
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

MODEL_PATH   = os.path.join(BASE_DIR, "models", "signal_generator", "rf_signal_model.pkl")
FEATURES_PAT = os.path.join(BASE_DIR, "data", "processed", "merged_features.csv")
OUT_PATH     = os.path.join(BASE_DIR, "data", "processed", "signals_latest.csv")

LOOKBACK_DAYS = 14
BUY_THRESH  = float(os.environ.get("BUY_THRESH", 0.60))
SELL_THRESH = float(os.environ.get("SELL_THRESH", 0.40))

def load_inputs():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(FEATURES_PAT):
        raise FileNotFoundError(f"Merged feature file not found: {FEATURES_PAT}")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURES_PAT)
    if "date" not in df.columns or "symbol" not in df.columns:
        raise ValueError("Expected 'date' and 'symbol' in merged_features.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    return model, df

def feature_columns(df: pd.DataFrame):
    # Text + market features
    cols = ["vader_compound"] + [c for c in df.columns if c.endswith("_tension")]
    cols += [c for c in df.columns if c.startswith(("ret_", "vol_", "ma_"))]
    return cols

def pick_latest_with_rows(df: pd.DataFrame):
    latest = df["date"].max()
    if pd.isna(latest):
        return None
    if not df[df["date"] == latest].empty:
        return latest
    for k in range(1, LOOKBACK_DAYS + 1):
        cand = latest - timedelta(days=k)
        if not df[df["date"] == cand].empty:
            return cand
    return None

def prob_to_signal(p: float) -> str:
    if p >= BUY_THRESH:  return "BUY"
    if p <= SELL_THRESH: return "SELL"
    return "HOLD"

def main():
    model, df = load_inputs()
    base_feats = feature_columns(df)

    scoring_date = pick_latest_with_rows(df)
    if scoring_date is None:
        print("No rows found in merged_features over the last 14 days.")
        raise SystemExit(0)

    df_day = df[df["date"].dt.date == scoring_date.date()].copy()
    if df_day.empty:
        print("Selected scoring date has no rows after date-only match.")
        raise SystemExit(0)

    df_day[base_feats] = df_day[base_feats].fillna(0.0)

    scorable = df_day[["symbol"] + base_feats].groupby("symbol", as_index=False).mean()

    sym_dum = pd.get_dummies(scorable["symbol"], prefix="sym")
    scorable = pd.concat([scorable, sym_dum], axis=1)

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = base_feats + [c for c in scorable.columns if c.startswith("sym_")]

    X = scorable.reindex(columns=expected, fill_value=0.0)

    try:
        proba = model.predict_proba(X)
        p_up = proba[:, 1]
    except Exception:
        y_pred = model.predict(X)
        p_up = (y_pred == 1).astype(float)

    signals = pd.DataFrame({
        "date": scoring_date.date().isoformat(),
        "symbol": scorable["symbol"],
        "p_up": p_up,
    })
    signals["signal"] = signals["p_up"].apply(prob_to_signal)
    signals["confidence"] = (signals["p_up"] - 0.5).abs() * 2  # 0..1

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    signals[["date","symbol","signal","p_up","confidence"]].to_csv(OUT_PATH, index=False)

    tag = "(latest)" if scoring_date == df["date"].max() else "(fallback)"
    print(f"📆 Date scored: {scoring_date.date()} {tag}")
    print("📈 Per-ticker signals:")
    for _, r in signals.iterrows():
        print(f"  • {r['symbol']}: {r['signal']} (p_up {r['p_up']:.2f}, conf {r['confidence']:.2f})")
    print(f"\nSaved to {OUT_PATH}")

if __name__ == "__main__":
    main()
