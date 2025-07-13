# Debt Call Analysis Dashboard

A Streamlit dashboard for analyzing and predicting outcomes of debt collection calls using machine learning and NLP.

## Features

- **Data Ingestion:** Load and preprocess raw call transcripts.
- **Sentiment Analysis:** Add sentiment scores to call transcripts.
- **Feature Engineering:** Extract features for ML models.
- **ML Predictions:** Predict payment outcomes using trained models.
- **Interactive Dashboard:** Visualize data and predictions with Streamlit.
- **Dummy Data Generation:** Generate synthetic call data for testing.

## Project Structure

```
dashboard/
    app.py                  # Streamlit dashboard app
data/
    processed_calls.csv     # Enriched data for ML
    raw_calls.csv           # Raw call transcripts
models/
    classifier.pkl          # Trained ML model
    label_encoder.pkl       # Label encoder for outcomes
notebooks/
    eda.ipynb               # Exploratory data analysis
scripts/
    generate_dummy_calls.py # Script to generate dummy data
src/
    feature_engineering.py
    ingestion.py
    model.py
    sentiment_analysis.py
```

## Setup

1. **Clone the repository and navigate to the project folder:**
    ```bash
    cd debt_call_analysis_dashboard
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **(Optional) Generate dummy data:**
    ```bash
    python scripts/generate_dummy_calls.py
    ```

5. **Run the Streamlit dashboard:**
    ```bash
    streamlit run dashboard/app.py
    ```

## Notes

- Make sure `data/raw_calls.csv` exists before running the dashboard. Use the dummy data script if needed.
- The dashboard expects trained models in the `models/` directory.

---
