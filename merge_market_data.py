import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    headlines_path = os.path.join(base_dir, "data", "processed", "headlines_with_sentiment.csv")
    prices_path = os.path.join(base_dir, "data", "raw", "stock_prices.csv")
    tension_path = os.path.join(base_dir, "data", "processed", "satellite_tension_scores.csv")
    output_path = os.path.join(base_dir, "data", "processed", "merged_features.csv")

    df_news = pd.read_csv(headlines_path)
    df_prices = pd.read_csv(prices_path)
    df_tension = pd.read_csv(tension_path) if os.path.exists(tension_path) else pd.DataFrame()

    df_news_grouped = df_news.groupby("date").agg({
        "vader_compound": "mean",
        "finbert_sentiment": lambda x: x.mode().iloc[0] if not x.mode().empty else "neutral"
    }).reset_index()

    df = pd.merge(df_prices, df_news_grouped, on="date", how="left")
    if not df_tension.empty:
        df = pd.merge(df, df_tension, on="date", how="left")

    df.to_csv(output_path, index=False)
    print(f"Saved merged feature set to {output_path}")
