# src/ingestion.py

import pandas as pd

def load_calls_data(path: str = "./data/raw_calls.csv") -> pd.DataFrame:
    """
    Load and preprocess the raw debt collection call data.

    Args:
        path (str): CSV file path to read from.

    Returns:
        pd.DataFrame: Cleaned dataframe ready for analysis.
    """
    try:
        df = pd.read_csv(path)

        # Strip extra whitespace in strings
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Fill empty fields
        df.fillna("", inplace=True)

        # Column sanity check
        required = ["call_id", "agent_id", "customer_id", "call_duration_sec", "transcript", "payment_outcome"]
        if not all(col in df.columns for col in required):
            raise ValueError(f"Missing one of required columns: {required}")

        print(f"✅ Loaded {len(df)} call transcripts.")
        return df

    except Exception as e:
        print(f"❌ Error loading calls: {e}")
        return pd.DataFrame()
