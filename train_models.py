"""
Train a ticker-aware classifier using text + market features, with optional tuning.
- Reads features from:  data/processed/merged_features.csv
- Reads labels  from:   data/processed/labeled_features.csv
- Writes model  to:     models/signal_generator/rf_signal_model.pkl
- Writes metrics to:    models/signal_generator/metrics_per_symbol.csv
Env toggles:
  USE_THRESHOLD_LABELS=1     -> recompute labels from forward returns with thresholds
  FUT_H=1                     -> forward horizon in days for label (default 1)
  FUT_RET_THRESH=0.003        -> threshold for BUY/SELL (default 0.3%)
  TUNE_RF=1                   -> run RandomizedSearchCV with TimeSeriesSplit
"""

import os
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import joblib

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MERGED_PATH  = os.path.join(BASE_DIR, "data", "processed", "merged_features.csv")
LABELED_PATH = os.path.join(BASE_DIR, "data", "processed", "labeled_features.csv")
MODEL_OUT    = os.path.join(BASE_DIR, "models", "signal_generator", "rf_signal_model.pkl")
METRICS_OUT  = os.path.join(BASE_DIR, "models", "signal_generator", "metrics_per_symbol.csv")
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

USE_THRESHOLD_LABELS = os.environ.get("USE_THRESHOLD_LABELS", "0") == "1"
FUT_H         = int(os.environ.get("FUT_H", "1"))                # forward horizon in days
FUT_RET_TH    = float(os.environ.get("FUT_RET_THRESH", "0.003")) # 0.3%
TUNE_RF       = os.environ.get("TUNE_RF", "0") == "1"

print(f"Loading features from: {MERGED_PATH}")
print(f"Loading labels   from: {LABELED_PATH}")

df_feat = pd.read_csv(MERGED_PATH)
df_lab  = pd.read_csv(LABELED_PATH)

for d in ("date",):
    if d in df_feat.columns: df_feat[d] = pd.to_datetime(df_feat[d], errors="coerce").dt.date
    if d in df_lab.columns:  df_lab[d]  = pd.to_datetime(df_lab[d],  errors="coerce").dt.date

def _norm_sym(s): return s.astype(str).str.strip().str.upper()
df_feat["symbol"] = _norm_sym(df_feat["symbol"])
df_lab["symbol"]  = _norm_sym(df_lab["symbol"])

text_feats = ["vader_compound"] + [c for c in df_feat.columns if c.endswith("_tension")]
mkt_feats  = [c for c in df_lab.columns  if c.startswith(("ret_", "vol_", "ma_"))]

missing_text = [c for c in text_feats if c not in df_feat.columns]
if missing_text:
    raise ValueError(f"merged_features.csv missing required NLP feature columns: {missing_text}")

for k in ("date","symbol"):
    if k not in df_feat.columns or k not in df_lab.columns:
        raise ValueError(f"Cannot merge: '{k}' missing in one of the files.")
    
keep_lab = ["date","symbol","adj_close"] if "adj_close" in df_lab.columns else ["date","symbol"]
label_candidates = ["target_label","label","target","signal","class","direction","y"]
have_labels = [c for c in label_candidates if c in df_lab.columns]
keep_lab += have_labels + mkt_feats

df = pd.merge(
    df_feat[["date","symbol"] + text_feats + (["adj_close"] if "adj_close" in df_feat.columns else [])],
    df_lab[keep_lab].drop_duplicates(["date","symbol"]),
    on=["date","symbol"],
    how="inner",
)

def compute_forward_labels(frame: pd.DataFrame, horizon=1, thresh=0.003) -> pd.Series:
    """
    BUY if fwd_return > +thresh, SELL if fwd_return < -thresh, else HOLD (NaN -> dropped).
    """
    if "adj_close" not in frame.columns:
        raise ValueError("adj_close is required to compute threshold labels.")
    tmp = frame.sort_values(["symbol","date"], kind="mergesort").copy()
    fwd_px = tmp.groupby("symbol")["adj_close"].shift(-horizon)
    fwd_ret = (fwd_px / tmp["adj_close"]) - 1.0
    out = pd.Series(index=tmp.index, dtype="object")
    out[fwd_ret >  +thresh] = "BUY"
    out[fwd_ret <  -thresh] = "SELL"

    return out.reindex(frame.index)

def map_existing_labels_to_binary(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.strip().str.lower()
    pos = {"buy","long","up","positive","bull","increase","1","true"}
    neg = {"sell","short","down","negative","bear","decrease","0","false"}
    out = pd.Series(np.nan, index=ss.index, dtype="float")
    out.loc[ss.isin(pos)] = 1.0
    out.loc[ss.isin(neg)] = 0.0

    return out

if USE_THRESHOLD_LABELS:
    print(f"Using threshold-based labels: horizon={FUT_H}d, thresh={FUT_RET_TH:.4f}")
    label_str = compute_forward_labels(df, horizon=FUT_H, thresh=FUT_RET_TH)
else:
    label_col = next((c for c in label_candidates if c in df.columns), None)
    if label_col is None:
        print("No explicit label column found; falling back to threshold-based labels.")
        label_str = compute_forward_labels(df, horizon=FUT_H, thresh=FUT_RET_TH)
    else:
        print(f"Using label column: {label_col}")
        label_str = df[label_col]

df[text_feats] = df[text_feats].fillna(0.0)
if mkt_feats:
    df[mkt_feats] = df[mkt_feats].fillna(0.0)

sym_dummies = pd.get_dummies(df["symbol"], prefix="sym")
df = pd.concat([df, sym_dummies], axis=1)

feature_cols = text_feats + mkt_feats + list(sym_dummies.columns)

y_float = map_existing_labels_to_binary(label_str)
train_df = pd.concat([df[["date","symbol"]], df[feature_cols], y_float.rename("label")], axis=1)
before = len(train_df)
train_df = train_df.dropna(subset=["label"])
after = len(train_df)
print(f"Kept {after} rows (dropped {before - after} rows with missing label).")
if after == 0:
    raise RuntimeError("No rows left after label filtering. Check label generation/mapping.")

train_df = train_df.sort_values("date", kind="mergesort")
split_idx = int(len(train_df) * 0.8)
train_part = train_df.iloc[:split_idx].copy()
test_part  = train_df.iloc[split_idx:].copy()

X_train = train_part[feature_cols]
y_train = train_part["label"].astype(int)
X_test  = test_part[feature_cols]
y_test  = test_part["label"].astype(int)

base_clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
    min_samples_leaf=3,
)

if TUNE_RF:
    print("Running RandomizedSearchCV with TimeSeriesSplit...")
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [2, 4, 6],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_grid,
        n_iter=10,
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    clf = search.best_estimator_
    print(f"Best params: {search.best_params_}")
else:
    clf = base_clf
    clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test), digits=4, zero_division=0))

per_symbol_rows = []
for sym, g in test_part.groupby("symbol"):
    if len(g) < 5: 
        continue
    y_true = g["label"].astype(int)
    y_pred = clf.predict(g[feature_cols])
    report = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
    row = {
        "symbol": sym,
        "support_0": int(report.get("0", {}).get("support", 0)),
        "precision_0": report.get("0", {}).get("precision", np.nan),
        "recall_0":    report.get("0", {}).get("recall", np.nan),
        "f1_0":        report.get("0", {}).get("f1-score", np.nan),
        "support_1": int(report.get("1", {}).get("support", 0)),
        "precision_1": report.get("1", {}).get("precision", np.nan),
        "recall_1":    report.get("1", {}).get("recall", np.nan),
        "f1_1":        report.get("1", {}).get("f1-score", np.nan),
        "accuracy":    report.get("accuracy", np.nan),
    }
    per_symbol_rows.append(row)
    print(f"\n=== Per-ticker report: {sym} ===")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

if per_symbol_rows:
    pd.DataFrame(per_symbol_rows).to_csv(METRICS_OUT, index=False)
    print(f"Per-ticker metrics saved to {METRICS_OUT}")
else:
    print("Per-ticker metrics not saved (not enough samples per symbol in test slice).")

joblib.dump(clf, MODEL_OUT)
print(f"✅ Model saved to {MODEL_OUT}")