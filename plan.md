Forecast & Risk Dashboard — Development Plan with File Structure

> *Goal:* Build the Forecast & Risk Dashboard quickly and realistically for the DEGA Hackathon using a clean modular file structure, minimal dependencies, and end-to-end reproducibility. No mocks — only real APIs, backtesting, and working code.

---

## 1 — Recommended Tech Stack

- *Backend:* Python (FastAPI)  
- *DB / Storage:* PostgreSQL + TimescaleDB (for time-series)  
- *ML Forecasting:* Statsmodels (ARIMA/Prophet optional), Scikit-learn (volatility modeling), SHAP (explainability)  
- *Frontend:* Next.js + Tailwind + Recharts/Plotly (charts)  
- *Treasury Integrations:*  
  - CoinGecko (price/time-series)  
  - Gnosis Safe Transaction Service API (treasury balances)  
  - Midnight.js (privacy-focused transactions)  

---

## 2 — File Structure

```bash
forecast-risk-dashboard/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entrypoint
│   │   ├── api/
│   │   │   ├── forecasts.py     # Endpoints for forecasts
│   │   │   ├── balances.py      # Treasury balances
│   │   │   └── hedge.py         # Hedging suggestions
│   │   ├── services/
│   │   │   ├── ingestion.py     # Data ingestion (CoinGecko, Safe)
│   │   │   ├── forecasting.py   # ARIMA/volatility models
│   │   │   ├── explainability.py# SHAP / rationale generator
│   │   │   └── hedge.py         # Policy-based hedge engine
│   │   ├── db/
│   │   │   ├── connection.py    # TimescaleDB connection
│   │   │   └── schema.sql       # DB schema (prices, balances, txs)
│   │   └── utils/
│   │       └── config.py        # Env vars, API keys
│   ├── tests/
│   │   ├── test_forecasting.py
│   │   ├── test_ingestion.py
│   │   └── test_hedge.py
│   └── requirements.txt
│
├── frontend/
│   ├── pages/
│   │   ├── index.tsx            # Landing dashboard
│   │   ├── forecast.tsx         # Forecast visualization
│   │   ├── hedge.tsx            # Hedge suggestion + simulate
│   ├── components/
│   │   ├── Chart.tsx            # Recharts/Plotly graph
│   │   ├── Card.tsx             # UI card component
│   │   └── Narrative.tsx        # AI rationale text block
│   ├── styles/
│   └── package.json
│
├── notebooks/
│   ├── backtest.ipynb           # Walk-forward validation notebook
│   └── prototype_forecast.ipynb # Fast iteration before API
│
├── infra/
│   ├── docker-compose.yml       # DB + backend + frontend
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── .env.example
│
├── README.md
└── PLAN.md                      # (this document)