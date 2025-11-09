import requests
import pandas as pd
from datetime import datetime

NEWSAPI_KEY = "a34774d8a510431c99601b408a5a95d6"
GNEWS_KEY = "085deecafffbcb6bca9f7a1f1f764ddd"
MEDIASTACK_KEY = "418841e29d88f058455e124a78e7a60c"


def fetch_newsapi():
    url = (
        f"https://newsapi.org/v2/everything?q=defense+OR+military+OR+Taiwan"
        f"&language=en&sortBy=publishedAt&pageSize=20&apiKey={NEWSAPI_KEY}"
    )
    response = requests.get(url)
    articles = response.json().get("articles", [])
    date = datetime.today().strftime('%Y-%m-%d')
    data = [
        {
            "source": "NewsAPI",
            "title": a["title"],
            "url": a["url"],
            "date": date
        }
        for a in articles if a.get("title")
    ]
    return pd.DataFrame(data)


def fetch_gnews():
    url = (
        f"https://gnews.io/api/v4/search?q=military+OR+defense+OR+China+Taiwan"
        f"&lang=en&country=us&max=20&token={GNEWS_KEY}"
    )
    response = requests.get(url)
    articles = response.json().get("articles", [])
    date = datetime.today().strftime('%Y-%m-%d')
    data = [
        {
            "source": "GNews",
            "title": a["title"],
            "url": a["url"],
            "date": date
        }
        for a in articles if a.get("title")
    ]
    return pd.DataFrame(data)


def fetch_mediastack():
    url = (
        f"http://api.mediastack.com/v1/news?access_key={MEDIASTACK_KEY}"
        f"&keywords=defense,military,Taiwan,China&languages=en&limit=20"
    )
    response = requests.get(url)
    articles = response.json().get("data", [])
    date = datetime.today().strftime('%Y-%m-%d')
    data = [
        {
            "source": "Mediastack",
            "title": a["title"],
            "url": a["url"],
            "date": date
        }
        for a in articles if a.get("title")
    ]
    return pd.DataFrame(data)


if __name__ == "__main__":
    df_list = []
    try:
        df_list.append(fetch_newsapi())
        print("✅ NewsAPI loaded")
    except Exception as e:
        print(f"❌ NewsAPI error: {e}")

    try:
        df_list.append(fetch_gnews())
        print("✅ GNews loaded")
    except Exception as e:
        print(f"❌ GNews error: {e}")

    try:
        df_list.append(fetch_mediastack())
        print("✅ Mediastack loaded")
    except Exception as e:
        print(f"❌ Mediastack error: {e}")

    if df_list:
        df_all = pd.concat(df_list)
        df_all.drop_duplicates(subset=["title", "url"], inplace=True)
        df_all.to_csv("data/raw/multisource_headlines.csv", index=False)
        print(df_all.head())
        print("📝 Headlines saved to data/raw/multisource_headlines.csv")
    else:
        print("⚠️  No data collected from any source.")