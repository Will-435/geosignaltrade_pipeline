import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

vader = SentimentIntensityAnalyzer()
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def apply_vader(df):
    df["vader_compound"] = df["title"].apply(lambda x: vader.polarity_scores(str(x))["compound"])
    return df

def apply_finbert(df):
    sentiments = []
    for text in df["title"]:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**tokens)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            label = torch.argmax(probs).item()
            sentiments.append(["neutral", "positive", "negative"][label])
    df["finbert_sentiment"] = sentiments
    return df
