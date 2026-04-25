# Beyond-VaR-Modeling-Post-Loss-Recovery-Duration-with-Survival-Analysis

A risk analytics project that combines Value at Risk (VaR) and Survival Analysis to study both how much an asset can lose and how long it takes to recover after an abnormal loss event.

The current implementation includes a complete VaR analysis pipeline for SPY, QQQ, GLD, and VOO, plus a Streamlit dashboard that exports violation-event data for the Survival Analysis teammate.

## Research Objective

This project studies two linked risk questions:

1. How large are one-day lower-tail losses under different Value at Risk specifications?
2. After an abnormal loss event occurs, how long does the asset take to recover?

The VaR layer identifies statistically meaningful loss events. The survival-analysis layer then treats those events as duration observations, with VIX included as a market-stress covariate.

## Scope

- Assets: SPY, QQQ, GLD, VOO
- Covariate: VIX (`^VIX`) stored separately
- Data source: `yfinance`
- Sample: 2015-01-01 through 2025-12-31
- Return frequency: daily adjusted-close percentage returns
- VaR confidence: 95%
- Rolling estimation window: 250 trading days
- Monte Carlo simulations: 10,000 per forecast date, random seed 42

## Methodology and Formulae

Daily returns are computed from adjusted close prices:

```text
r_t = P_t / P_{t-1} - 1
```

For a confidence level `c`, the lower-tail probability is:

```text
alpha = 1 - c
```

The project estimates one-day-ahead VaR using only information available before the realized return. With a 250-trading-day rolling window:

```text
Historical VaR:
VaR^{HS}_{t,alpha} = Q_alpha(r_{t-250}, ..., r_{t-1})

Parametric Normal VaR:
VaR^{N}_{t,alpha} = mu_hat_{t-1,250} + sigma_hat_{t-1,250} * Phi^{-1}(alpha)

Monte Carlo VaR:
r^{(j)}_t ~ N(mu_hat_{t-1,250}, sigma_hat^2_{t-1,250})
VaR^{MC}_{t,alpha} = Q_alpha({r^{(j)}_t}_{j=1}^{10000})
```

A VaR violation occurs when the realized return is below the forecast VaR threshold:

```text
I_t = 1{r_t < VaR_{t,alpha}}
```

Backtesting uses Kupiec's unconditional coverage test and Christoffersen's conditional coverage test:

```text
LR_uc = -2 log [ ((1-alpha)^(n-x) alpha^x) / ((1-p_hat)^(n-x) p_hat^x) ] ~ chi-square(1)
p_hat = x / n

LR_cc = LR_uc + LR_ind ~ chi-square(2)
```

Kupiec evaluates whether the average violation rate matches the theoretical tail probability. Christoffersen adds an independence test, so a model can pass Kupiec but fail Christoffersen when violations cluster during stressed regimes.

## Empirical Interpretation

The return diagnostics reject normality for all four assets at the 5% level using Jarque-Bera and D'Agostino K2 tests. This matters because VaR exceptions are lower-tail events: skewness, excess kurtosis, and volatility clustering can materially affect violation timing even when the unconditional violation rate is close to the target level.

In the 95% baseline:

- SPY and VOO show strong excess kurtosis, consistent with crisis-period tail concentration.
- QQQ has a higher violation rate under Parametric Normal and Monte Carlo VaR than the theoretical 5% target, indicating more aggressive lower-tail realization than the Gaussian rolling model expects.
- GLD has lower equity-market beta but still rejects normality, so it should not be treated as a normally distributed hedge asset.
- VIX is retained as a separate covariate to explain whether violation events start during elevated market-stress states.

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

The dashboard also includes an academic methodology section with VaR and backtesting formulae, plus a dynamic empirical interpretation panel covering distribution diagnostics, violation severity, and VIX conditions at violation dates.

### Public Access

The repo is ready for Streamlit Community Cloud:

1. Go to https://share.streamlit.io
2. Create a new app from this GitHub repository.
3. Select branch `main`.
4. Use `streamlit_app.py` as the app entrypoint.

Streamlit Cloud will install `requirements.txt` and serve the dashboard from the processed CSV files committed in `data/processed`.

## Modeling Note

VaR forecasts for day `t` are estimated with the prior 250 trading-day window and shifted forward one day before comparing against the realized return. This avoids look-ahead bias in the violation checks.
