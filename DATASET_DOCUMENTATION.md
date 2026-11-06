# InvestWise AI dataset documentation

## Source and classification

The application uses rows from its local SQLite `transactions` table. No Kaggle, UCI, or other public dataset is currently included. On first run, the application seeds a reproducible synthetic demonstration ledger; rows later entered in the Transactions page are user-entered. Dataset Explorer determines whether the currently loaded rows are synthetic, user-entered, or mixed from the saved descriptions. The explorer reports all counts and date ranges from the current database rather than hardcoded totals.

## Synthetic data generation

`database.py:seed_demo` uses NumPy random generator seed 27 to generate twelve monthly periods ending in the current month. Salary begins at INR 75,000 and rises by INR 500 each month. Freelance income is an integer drawn from the half-open range INR 4,000–12,000. Each month has one expense row per category: Food, Shopping, Transportation, Education, Entertainment, Healthcare, Rent, Utilities, and Subscriptions. Non-rent category amounts are drawn from category-specific normal distributions and floored at INR 200; Rent is fixed at INR 15,000. Records are placed on the first day of each month, with salary on day one and freelance income on day 18. Budgets are 118% of each category's configured base expense amount. Four fictional holdings and synthetic linearly interpolated asset price histories are also seeded.

Synthetic records make the demo reproducible, immediately usable, and safe to present without exposing personal financial information. The generator intentionally provides multiple income sources, spending categories with different scales and variability, and a simple income trend. It does not simulate real household behavior: transaction timing is highly regular, each month has only one expense per category, rent never varies, no external market prices are used, and the short series has no realistic seasonality or economic shocks. The category normal draws are not calibrated from a real population. Consequently the data cannot establish real-world forecast quality, fraud detection performance, or representative financial behavior. Reset Demo Data recreates the sample ledger; it does not overwrite existing data during normal startup.

## Raw transaction columns

| Field | Description |
|---|---|
| `id` | SQLite row identifier |
| `date` | Transaction date, stored as ISO date text |
| `kind` | Income or Expense |
| `category` | User-selected transaction category |
| `amount` | Positive transaction amount, INR in the demo |
| `description` | User text or synthetic-demo marker |

## LSTM dataset and target

The source rows are grouped by calendar month into `income` and `expenses` totals. Missing calendar months between the earliest and latest transaction are inserted with zero totals. The target is the next monthly `expenses` value. The model uses the preceding sliding sequence of up to six monthly expense totals; `income` is displayed for dataset context but is not currently an LSTM input feature. Window length is `min(6, max(2, number_of_months // 3))`. `MinMaxScaler` is fitted only to training target observations and applied to the ordered windows. Target windows are split chronologically: approximately 65% train, the next 17% validation, and the remaining later windows test. Validation windows may use earlier observed months as context, which is available at forecast time; their target months never enter the training-target split. The model selects an epoch using validation loss, is refit on train plus validation using that epoch count, then evaluated on the held-out test windows. The displayed naive comparator predicts the last observed value in each test window. Dataset Explorer reports actual target date ranges and provides searchable, paginated previews; column headers in the preview are sortable.

## Autoencoder features and target

The Autoencoder filters to positive, valid Expense records and sorts them by date and then transaction ID. Its four input features are `amount_log1p` (natural log of one plus amount), `day_of_month`, `weekday` (Monday=0), and `category_index` (alphabetical ordinal encoding over the currently loaded categories). The first 80% of expense rows train a `StandardScaler` and the model; the later rows are used for validation. The category vocabulary is derived from the loaded dataset, so the ordinal category values are dataset-dependent and should not be compared across different ledgers. Reconstruction mean squared error is computed for all usable rows. The threshold is the 95th percentile of training-row reconstruction errors. A row above that threshold is flagged for human review, not classified as fraud. The Autoencoder export includes the raw engineered values and the standardized model inputs so the values supplied to the network can be identified directly.

## Reproducibility and limitations

The demo data random seed is 27 and model initialization uses TensorFlow/Keras seed 27. Reset the demo data to restore the example. Model training is explicit and does not occur on app startup. Trained model artifacts and run summaries are stored locally under `deep_learning/saved_models` and are reloaded only when the current transaction fingerprint matches. Local storage can be ephemeral in hosted deployments. The synthetic time series is short; its forecasts and anomaly flags are not external benchmark results and should not be generalized as financial guidance.
