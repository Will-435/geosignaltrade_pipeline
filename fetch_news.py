import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_scraper import (
    fetch_newsapi,
    fetch_gnews,
    fetch_mediastack
)

if __name__ == "__main__":
    df_list = []

    try:
        df_newsapi = fetch_newsapi()
        print(f"✅ NewsAPI returned {len(df_newsapi)} articles")
        df_list.append(df_newsapi)
    except Exception as e:
        print(f"❌ NewsAPI error: {e}")

    try:
        df_gnews = fetch_gnews()
        print(f"✅ GNews returned {len(df_gnews)} articles")
        df_list.append(df_gnews)
    except Exception as e:
        print(f"❌ GNews error: {e}")

    try:
        df_mediastack = fetch_mediastack()
        print(f"✅ Mediastack returned {len(df_mediastack)} articles")
        df_list.append(df_mediastack)
    except Exception as e:
        print(f"❌ Mediastack error: {e}")

    if df_list:
        df_all = pd.concat(df_list)
        df_all.drop_duplicates(subset=["title", "url"], inplace=True)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "data", "raw", "multisource_headlines.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not df_all.empty:
            df_all.to_csv(output_path, index=False)
            print(f"📝 Headlines saved to {output_path}")
        else:
            print("⚠️ No headlines to save. The file was not written.")
    else:
        print("⚠️ All API scrapers failed — no data saved.")