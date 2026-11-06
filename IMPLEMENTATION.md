# Implementation overview

## Architecture

Streamlit `app.py` owns navigation and presentation. `database.py` initializes the SQLite schema and offers parameterized data access helpers. The UI modules in the main app call those helpers for transaction, budget, investment, profile, and report flows. Plotly renders financial visualizations.

## Dataset pipeline

`deep_learning/preprocessing.py` derives monthly income/expense rows, chronological LSTM sliding windows and splits, and sorted transaction-level Autoencoder features. Dataset Explorer uses the same builders as training. `DATASET_DOCUMENTATION.md` describes the source and feature meanings.

## Neural networks

`deep_learning/models.py` trains a two-layer LSTM forecaster and a dense bottleneck Autoencoder with Keras. The LSTM evaluates on later held-out windows and reports actual MAE/RMSE/MAPE alongside a last-value baseline. The Autoencoder trains chronologically on standardized expense features, validates on later rows, and flags reconstruction errors above the training 95th percentile. Model files and metadata are optionally cached locally by `deep_learning/model_store.py`, guarded by a current-data fingerprint.

## Multi-agent system

`agents.py` contains spending, savings, budget, portfolio, health, market-mark, risk-indicator, learning-status, explanation, and notification calculations. The Decision Coordinator combines those measured outputs with trained LSTM and Autoencoder results when current-fingerprint artifacts exist. Each generated finding includes supporting data, calculation reasoning, a review action, and a limitation. The market agent explicitly treats prices as manually entered marks; risk figures are descriptive and do not forecast loss. Unusual autoencoder records are not labeled as fraud.

## Frontend and safeguards

The Streamlit application has Dashboard, Dataset Explorer, Transactions, Budget Planner, Investment Portfolio, AI Financial Insights, Deep Learning Lab, AI Financial Assistant, Reports, and Settings pages. A shared dark navy/emerald design system styles the interface. Dataset previews expose actual schema/statistics, chronological ranges, and searchable, paginated tables whose columns can be sorted. CSV downloads contain the source ledger, monthly LSTM rows, and raw plus standardized Autoencoder inputs. SQLite writes use placeholders; transaction amounts, budgets, quantities, and prices are constrained to positive values by UI validation and schema checks. The data and model results are local educational examples, not financial advice.
