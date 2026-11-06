# INVESTWISE AI

An academic, local financial management and decision support dashboard built with Streamlit, SQLite, Plotly, scikit-learn, and TensorFlow/Keras. The demo database is seeded on first launch with reproducible fictional records.

## Problem and objectives

People often lack one place to review spending, budgets, and investment holdings. InvestWise AI demonstrates personal finance tracking, explainable financial indicators, and neural network applications to financial sequences and outlier detection. It is an educational prototype, not an investment adviser.

## Architecture

- `app.py`: Streamlit navigation, dashboard, CRUD flows, insights, assistant, and reports.
- `database.py`: SQLite schema, CRUD helpers, and reproducible demo seed.
- `deep_learning/models.py`: Keras LSTM forecasting and autoencoder training, preprocessing, and metrics.
- `deep_learning/preprocessing.py`: shared dataset aggregations, chronological windows, and model features.
- `deep_learning/model_store.py`: fingerprint-checked local model and run metadata persistence.
- `agents.py`: spending, savings, budget, portfolio, and health agents plus the Decision Coordinator.
- `data/investwise.db`: persistent local SQLite database created at runtime.

The application uses ten sidebar sections: Dashboard, Dataset Explorer, Transactions, Budget Planner, Investment Portfolio, AI Financial Insights, Deep Learning Lab, AI Financial Assistant, Reports, and Settings.

## Deep learning

The LSTM forecasts monthly expense totals from ordered monthly observations. It MinMax-scales based on training targets, creates sliding windows, splits target windows chronologically into train, validation, and held-out test sets, and reports MAE, RMSE, and MAPE from the held-out test predictions. Its architecture uses 64 and 32 unit LSTM layers, dropout, and dense layers.

The dense autoencoder learns expense feature representations (log amount, day of month, weekday, and category). It trains on the earlier 80% of expense rows and validates on later rows; reconstruction error above the training-set 95th percentile is flagged for review. Both models train when requested in Deep Learning Lab. See `MODEL_RESULTS.md` for one recorded run and its limitations.

The Decision Coordinator combines actual transaction, budget, portfolio, and profile calculations. When models have been trained in the current session, it also incorporates the held-out LSTM evaluation/forecast and autoencoder anomaly counts.

## Setup

Python 3.11 is selected for the pinned deployment dependencies. From this directory:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app uses no external API or account. Keras model training is initiated by buttons, so startup does not retrain models. Model artifacts are stored locally and tied to a fingerprint of the transaction dataset. Edit, delete, budget, and portfolio changes persist to SQLite. `Reset demo data` replaces financial records with the reproducible fictional demo set.

Run functional checks with `python -m unittest discover -s tests -v`. Tests cover database CRUD, agent calculations, both trained models, and rendering of each sidebar page with Streamlit AppTest.

## Data and cautions

All seeded records are synthetic. Investment prices are explicit example prices and are never live quotes. The financial health indicator is an illustrative weighted score, not a certified assessment. Forecasts and anomaly flags are not financial advice. CSV downloads are available under Reports; PDF export is not included.
