# Deployment guide: Streamlit Community Cloud

## 1. Prepare a GitHub repository

Create an empty repository on GitHub. In this project directory, initialize Git if needed, inspect the staged files, then commit and push the source. `.gitignore` excludes the local SQLite database, model artifacts, secrets, virtual environments, and Python caches. Do not add credentials or private financial exports.

```bash
git init
git add .
git status
git commit -m "Prepare InvestWise AI deployment"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

## 2. Deploy

Sign in to Streamlit Community Cloud, choose **Create app**, select the repository and `main` branch, and set `app.py` as the application entry point. The pinned dependencies are installed from `requirements.txt`. No API key is needed.

## 3. Python and dependencies

Use Python 3.11. It is supported by the pinned TensorFlow 2.18.0 release and the other pinned packages. If the Cloud workspace offers a Python version setting, select 3.11. Local setup uses `python -m pip install -r requirements.txt` followed by `streamlit run app.py`.

## 4. Database and persistence limits

The app creates `data/investwise.db` at startup and seeds a fictional demo ledger only when the transaction table is empty. Reset is available in the sidebar. Streamlit Community Cloud local storage may be discarded on restart or redeploy; this project does not claim durable cloud persistence or multi-user isolation. For a multi-user deployment, connect a managed PostgreSQL service (for example, a hosted Postgres provider) and keep credentials in Streamlit secrets; database migrations, authentication, tenancy isolation, backups, and access controls would be required before using real financial records.

## 5. Model training and artifacts

Neural networks train only after an explicit action in Deep Learning Lab. Local `.keras`, scaler, and JSON run artifacts are ignored by Git and may not survive Cloud restarts. Retrain after deploy or data changes. The app checks the transaction fingerprint before reloading local artifacts. Short synthetic samples are educational only.

## 6. Troubleshooting

- **TensorFlow installation failure:** confirm Python 3.11 and use the pinned `tensorflow==2.18.0` dependency.
- **Cold start:** TensorFlow import takes longer than a simple Streamlit app; allow the first app health check to finish.
- **No charts or model output:** restore demo data from the sidebar or add enough records; LSTM needs at least six monthly observations and the Autoencoder needs eight valid expenses.
- **Model result not restored:** artifacts are local and fingerprint-bound; retrain from Deep Learning Lab.
- **Demo records reset after a restart:** this can happen with ephemeral hosted filesystems. Use a managed database for durable data.
