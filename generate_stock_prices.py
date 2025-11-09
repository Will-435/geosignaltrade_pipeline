import os
import sys
from datetime import datetime, timedelta
from datetime import UTC
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
_root_candidates = [
    _here,
    os.path.dirname(_here),
    os.path.dirname(os.path.dirname(_here)),
]
for cand in _root_candidates:
    if os.path.isdir(os.path.join(cand, "data")):
        sys.path.append(cand)
        PROJECT_ROOT = cand
        break
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(_here))
    sys.path.append(PROJECT_ROOT)

def _build_output_path():
    out = os.path.join(PROJECT_ROOT, "data", "raw", "stock_prices.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out

def _get_date_range():
    end = os.environ.get("STOCK_END_DATE")
    start = os.environ.get("STOCK_START_DATE")
    end_dt = pd.to_datetime(end).date() if end else datetime.now(UTC).date()
    start_dt = pd.to_datetime(start).date() if start else end_dt - timedelta(days=365*3)
    return start_dt, end_dt

def fetch_prices():
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance is required. Please `pip install yfinance`.") from e

    tickers = {
        "TSMC": "TSM",       
        "INTC": "INTC",       
        "SMSN": "005930.KS",  
    }

    start_dt, end_dt = _get_date_range()

    data = yf.download(
        list(tickers.values()),
        start=start_dt.isoformat(),
        end=(end_dt + timedelta(days=1)).isoformat(),  # inclusive end
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("No price data returned from yfinance. Check tickers or connectivity.")

    frames = []
    for label, y_ticker in tickers.items():
        df_sym = pd.DataFrame({"date": data.index})
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            key = (col, y_ticker) if isinstance(data.columns, pd.MultiIndex) else col
            df_sym[col.lower().replace(" ", "_")] = data[key].values if key in data.columns else pd.NA
        df_sym["symbol"] = label
        frames.append(df_sym)

    df_all = pd.concat(frames, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date.astype(str)

    cols = ["date", "symbol", "adj_close", "close", "open", "high", "low", "volume"]
    return df_all[[c for c in cols if c in df_all.columns]]

if __name__ == "__main__":
    df = fetch_prices()
    output = _build_output_path()
    df.to_csv(output, index=False)
    print(f"📈 Saved {len(df)} rows of stock prices to {output}")
