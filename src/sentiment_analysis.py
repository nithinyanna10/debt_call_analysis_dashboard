# src/sentiment_analysis.py

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

# Initialize once globally
vader = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> float:
    """
    Return compound sentiment score from VADER [-1, 1].
    """
    score = vader.polarity_scores(text)["compound"]
    return score

def add_sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentiment score column to a transcript dataframe.
    """
    df["sentiment_score"] = df["transcript"].apply(analyze_sentiment)
    return df
