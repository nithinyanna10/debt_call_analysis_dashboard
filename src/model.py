# src/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_model(path: str = "data/processed_calls.csv", model_output: str = "models/classifier.pkl"):
    """
    Train a classifier to predict payment outcome based on call features.

    Args:
        path (str): Path to processed CSV with features.
        model_output (str): Where to save the trained model.
    """
    # Load data
    df = pd.read_csv(path)

    # Encode target
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["payment_outcome"])  # Paid → 1, Not Paid → 0

    # Features
    features = [
        "sentiment_score",
        "objection_count",
        "positive_count",
        "negative_count",
        "resolution_keyword",
        "call_duration_sec"
    ]

    X = df[features]
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    print("✅ Classification Report:\n", classification_report(y_test, y_pred))
    print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model and label encoder
    joblib.dump(clf, model_output)
    joblib.dump(le, "models/label_encoder.pkl")

    print(f"📦 Model saved to {model_output}")

def predict_single(input_dict, model_path="models/classifier.pkl", encoder_path="models/label_encoder.pkl"):
    """
    Predict the outcome for a single call input (dict of features).
    """
    clf = joblib.load(model_path)
    le = joblib.load(encoder_path)

    input_df = pd.DataFrame([input_dict])
    pred = clf.predict(input_df)[0]
    label = le.inverse_transform([pred])[0]

    return label
