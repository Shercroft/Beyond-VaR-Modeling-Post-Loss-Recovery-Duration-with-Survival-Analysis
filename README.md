# Beyond-VaR-Modeling-Post-Loss-Recovery-Duration-with-Survival-Analysis

A risk analytics project that combines Value at Risk (VaR) and Survival Analysis to study both how much an asset can lose and how long it takes to recover after an abnormal loss event.

The current implementation includes a complete VaR analysis pipeline for SPY, QQQ, GLD, and VOO, plus a Streamlit dashboard that exports violation-event data for the Survival Analysis teammate.

## Scope

- Assets: SPY, QQQ, GLD, VOO
- Covariate: VIX (`^VIX`) stored separately
- Data source: `yfinance`
- Sample: 2015-01-01 through 2025-12-31
- Return frequency: daily adjusted-close percentage returns
- VaR confidence: 95%
- Rolling estimation window: 250 trading days
- Monte Carlo simulations: 10,000 per forecast date, random seed 42

## Outputs

The notebook writes reusable outputs to:

- `data/raw/adjusted_close_prices.csv`
- `data/raw/vix_adjusted_close.csv`
- `data/processed/daily_returns.csv`
- `data/processed/vix_aligned.csv`
- `data/processed/return_summary_statistics.csv`
- `data/processed/normality_tests.csv`
- `data/processed/rolling_var_estimates.csv`
- `data/processed/var_backtest_summary.csv`
- `outputs/figures/qq_plots_returns.png`
- `outputs/figures/*_var_vix.png`

## Run

From this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/01_var_analysis_pipeline.ipynb` and run all cells.

To execute the notebook non-interactively:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_var_analysis_pipeline.ipynb --output 01_var_analysis_pipeline_executed.ipynb
```

## Dashboard

Run the Streamlit dashboard from this project directory:

```bash
streamlit run dashboard.py
```

The dashboard lets you choose ticker, VaR method, confidence level, and date range. It shows summary backtest cards, a Plotly returns/VaR/VIX chart, yearly violation counts, Kupiec and Christoffersen test results, and a CSV export for survival-analysis violation events.

### Public Access

The repo is ready for Streamlit Community Cloud:

1. Go to https://share.streamlit.io
2. Create a new app from this GitHub repository.
3. Select branch `main`.
4. Use `streamlit_app.py` as the app entrypoint.

Streamlit Cloud will install `requirements.txt` and serve the dashboard from the processed CSV files committed in `data/processed`.

## Modeling Note

VaR forecasts for day `t` are estimated with the prior 250 trading-day window and shifted forward one day before comparing against the realized return. This avoids look-ahead bias in the violation checks.
