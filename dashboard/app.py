# dashboard/app.py (cleaned, no PDF, no emojis)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.model import predict_single
from src.ingestion import load_calls_data
from src.sentiment_analysis import add_sentiment_column
from src.feature_engineering import extract_features

st.set_page_config(page_title="Debt Collection Dashboard", layout="wide")
st.title("Debt Collection Call Analysis Dashboard")

# --- Load and preprocess data ---
@st.cache_data(show_spinner=False)
def get_data():
    st.cache_data.clear()
    df = load_calls_data("../data/raw_calls.csv")
    df = add_sentiment_column(df)
    df = extract_features(df)
    return df

df = get_data()

model = joblib.load("../models/classifier.pkl")
encoder = joblib.load("../models/label_encoder.pkl")

# --- Layout Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Visual Insights",
    "Predict Outcome",
    "Upload & Batch Predict",
    "KPI Dashboard"
])

# ------------------- Tab 1: Overview -------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    col1.metric("Total Calls", len(df))
    col2.metric("Success Rate", f"{(df['payment_outcome'] == 'Paid').mean() * 100:.1f}%")

# ------------------- Tab 2: Visual Insights -------------------
with tab2:
    st.subheader("Visual Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sentiment Distribution**")
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        sns.histplot(df["sentiment_score"], kde=True, ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.markdown("**Objection Frequency by Outcome**")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        sns.boxplot(data=df, x="payment_outcome", y="objection_count", ax=ax2)
        st.pyplot(fig2)

    st.markdown("**Agent-wise Payment Success Rate**")
    agent_success = df.groupby("agent_id")["payment_outcome"].value_counts().unstack().fillna(0)
    agent_success["SuccessRate"] = agent_success["Paid"] / agent_success.sum(axis=1)
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    agent_success["SuccessRate"].sort_values().plot(kind="barh", ax=ax3)
    st.pyplot(fig3)

    st.markdown("**Top 5 Objection Reasons**")
    objection_keywords = ["lost my job", "already paid", "need more time", "wrong number", "salary", "stop calling"]
    objection_counts = {kw: df["transcript"].str.lower().str.count(kw).sum() for kw in objection_keywords}
    top_objections = pd.Series(objection_counts).sort_values(ascending=False).head(5)
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    top_objections.plot(kind="bar", ax=ax4, color="orange")
    ax4.set_title("Top 5 Objection Reasons")
    st.pyplot(fig4)

# ------------------- Tab 3: Predict Outcome -------------------
with tab3:
    st.subheader("Predict Outcome for a Single Call")

    with st.form("predict_form"):
        sentiment = st.slider("Sentiment Score", -1.0, 1.0, 0.2, 0.01)
        objections = st.number_input("Objection Count", 0, 10, 1)
        positive = st.number_input("Positive Phrase Count", 0, 10, 1)
        negative = st.number_input("Negative Phrase Count", 0, 10, 1)
        resolution = st.selectbox("Resolution Keyword Present?", ["Yes", "No"])
        duration = st.slider("Call Duration (seconds)", 60, 600, 300)
        submitted = st.form_submit_button("Predict Outcome")

    if submitted:
        features = {
            "sentiment_score": sentiment,
            "objection_count": objections,
            "positive_count": positive,
            "negative_count": negative,
            "resolution_keyword": 1 if resolution == "Yes" else 0,
            "call_duration_sec": duration
        }

        pred = model.predict([list(features.values())])[0]
        proba = model.predict_proba([list(features.values())])[0][pred]
        label = encoder.inverse_transform([pred])[0]
        emoji = "✅" if label == "Paid" else "❌"
        color = "green" if label == "Paid" else "red"

        st.markdown(f"### Prediction: {emoji} **:{color}[{label}]**")
        st.markdown(f"Confidence: **{proba * 100:.2f}%**")

# ------------------- Tab 4: Upload & Batch Predict -------------------
with tab4:
    st.subheader("Upload CSV for Batch Prediction")
    uploaded = st.file_uploader("Upload a CSV with call features", type="csv")

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        required_cols = [
            "sentiment_score", "objection_count", "positive_count",
            "negative_count", "resolution_keyword", "call_duration_sec"
        ]

        if not set(required_cols).issubset(batch_df.columns):
            st.error("Uploaded file must include all required feature columns.")
        else:
            preds = model.predict(batch_df[required_cols])
            probas = model.predict_proba(batch_df[required_cols]).max(axis=1)
            labels = encoder.inverse_transform(preds)

            batch_df["Prediction"] = labels
            batch_df["Confidence"] = (probas * 100).round(2).astype(str) + "%"
            batch_df["Emoji"] = batch_df["Prediction"].apply(lambda x: "✅" if x == "Paid" else "❌")

            st.success("Predictions complete!")
            st.dataframe(batch_df.head(10))

            st.download_button(
                "Download CSV",
                batch_df.to_csv(index=False),
                file_name="predicted_calls.csv",
                mime="text/csv"
            )

# ------------------- Tab 5: KPI Dashboard -------------------
with tab5:
    st.subheader("Objection-to-Success Rate")
    objection_keywords = ["lost my job", "already paid", "need more time", "wrong number", "salary", "stop calling"]
    obj_df = df.copy()
    for kw in objection_keywords:
        obj_df[kw] = obj_df["transcript"].str.lower().str.contains(kw).astype(int)

    obj_summary = []
    for kw in objection_keywords:
        success = obj_df[obj_df[kw] == 1].groupby("payment_outcome").size()
        obj_summary.append({
            "Objection": kw,
            "Paid": success.get("Paid", 0),
            "Not Paid": success.get("Not Paid", 0)
        })

    obj_plot_df = pd.DataFrame(obj_summary).set_index("Objection")
    obj_plot_df.plot(kind="bar", stacked=False, figsize=(8, 4))
    st.pyplot(plt.gcf())

    st.subheader("Agent Resolution Ratio")
    resolution_ratio = df.groupby("agent_id")["resolution_keyword"].mean().sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    resolution_ratio.plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Resolution Ratio")
    st.pyplot(fig1)

    st.subheader("Avg Sentiment by Agent & Result")
    heat_data = df.groupby(["agent_id", "payment_outcome"])["sentiment_score"].mean().unstack()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(heat_data, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Time-of-Day Success Rate (Sample Placeholder)")
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    ax3.plot([9, 12, 15, 18, 21], [0.5, 0.55, 0.62, 0.58, 0.45], marker="o")
    ax3.set_xticks([9, 12, 15, 18, 21])
    ax3.set_title("Sample Success Trend by Hour")
    ax3.set_ylabel("Success Rate")
    st.pyplot(fig3)
