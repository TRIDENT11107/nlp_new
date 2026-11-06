# Model results

## Actual run

Trained on 2026-11-06 from the current local ledger. Dataset fingerprint prefix: `7db99a7dd356dc23`.

- Source: synthetic demo SQLite rows; 132 raw transactions across 12 monthly observations. The model metrics below were calculated on the local runtime’s installed packages (Python 3.13.3, TensorFlow 2.20.0). Deployment pins Python 3.11 / TensorFlow 2.18.0 and should be retrained there before reporting those deployment-specific numbers.
- LSTM: input shape `(None, 4, 1)`, 29,857 parameters; 5 train windows, 1 validation window, 2 held-out test windows (the last two target months in the recorded series). Adam optimizer, MSE loss, batch size 5, maximum 45 epochs, early stopping patience 8; validation selected epoch 41 for the final train-plus-validation fit. The target scaler was fitted on training targets only. Test MAE **5,647.97 INR**, RMSE **5,953.16 INR**, MAPE **11.48%**.
- Last-value baseline on the same test windows: MAE **1,832.24 INR**, RMSE **2,149.93 INR**, MAPE **3.83%**. On this run, the LSTM did **not** improve on the naive baseline.
- Recursive three-month LSTM forecast: **44,495.08**, **46,458.83**, and **45,995.80 INR**. These estimates are illustrative and the sample is too short for reliable forecasting.
- Autoencoder: 108 valid expense rows; chronological split 86 train / 22 validation; architecture 4 → 8 → 3 → 8 → 4 with **135 parameters**; Adam optimizer, MSE loss, batch size 16, maximum 35 epochs, early stopping patience 6. The StandardScaler was fit on the first 86 rows. The threshold was **1.25009** (95th percentile of training-row reconstruction MSE). **5 rows** were flagged across scored records.
- Flagged examples: Subscriptions INR 480.09 (2025-12-01), Healthcare INR 200.00 (2026-01-01), Subscriptions INR 418.35 (2026-02-01), Subscriptions INR 623.19 (2026-03-01), Subscriptions INR 327.95 (2026-06-01).

## Interpretation

These are actual outputs from one reproducible-seed code run, not a claim of model quality. Only two LSTM test windows exist; the test set is extremely small, the LSTM underperformed the naive comparator, and all seeded ledger records are synthetic. Autoencoder flags are reconstruction-error prompts, not fraud findings. Changing or resetting transactions invalidates this run; the UI checks a transaction fingerprint and hides stale artifacts. Artifacts were already present and recorded as complete at audit, so they were reused rather than needlessly retrained.
