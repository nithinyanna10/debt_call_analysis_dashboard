# src/feature_engineering.py

import pandas as pd
import re

# Sample objection patterns (can expand later)
OBJECTION_KEYWORDS = [
    "can't pay", "already paid", "need more time", "lost my job",
    "wrong number", "waiting for salary", "stop calling", "talk to lawyer"
]

POSITIVE_KEYWORDS = ["thank you", "sure", "yes", "I’ll pay", "confirmed"]
NEGATIVE_KEYWORDS = ["no", "not paying", "stop", "harassment"]

def count_keywords(text: str, keywords: list) -> int:
    pattern = "|".join(re.escape(kw) for kw in keywords)
    return len(re.findall(pattern, text.lower()))

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df["objection_count"] = df["transcript"].apply(lambda x: count_keywords(x, OBJECTION_KEYWORDS))
    df["positive_count"] = df["transcript"].apply(lambda x: count_keywords(x, POSITIVE_KEYWORDS))
    df["negative_count"] = df["transcript"].apply(lambda x: count_keywords(x, NEGATIVE_KEYWORDS))
    df["resolution_keyword"] = df["transcript"].str.contains("installment|process the payment|confirmed", case=False).astype(int)
    return df
