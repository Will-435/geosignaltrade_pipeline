import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nlp_helpers import apply_vader, apply_finbert

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "raw", "multisource_headlines.csv")
    output_path = os.path.join(base_dir, "data", "processed", "headlines_with_sentiment.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")

    df = pd.read_csv(input_path)
    df["title"] = df["title"].fillna("")

    df = apply_vader(df)
    df = apply_finbert(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved sentiment-analyzed headlines to {output_path}")
