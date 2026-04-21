from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


TICKERS = ["SPY", "QQQ", "GLD", "VOO"]
VAR_METHODS = ["Historical", "Parametric Normal", "Monte Carlo"]
METHOD_TO_COLUMN = {
    "Historical": "VaR_Historical",
    "Parametric Normal": "VaR_Normal",
    "Monte Carlo": "VaR_MonteCarlo",
}


def normalize_confidence(confidence_level: float) -> float:
    """Accept confidence as either 0.95 or 95 and return 0.95."""
    confidence = float(confidence_level)
    if confidence > 1:
        confidence /= 100
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must be between 0 and 1, or between 0 and 100.")
    return confidence


def violation_column(method: str) -> str:
    return f"Violation_{METHOD_TO_COLUMN[method].replace('VaR_', '')}"


def add_violation_flags(var_df: pd.DataFrame, methods: Iterable[str] = VAR_METHODS) -> pd.DataFrame:
    out = var_df.copy()
    for method in methods:
        column = METHOD_TO_COLUMN[method]
        out[violation_column(method)] = out["return"].lt(out[column]) & out[column].notna()
    return out


def monte_carlo_var(
    rolling_mean: pd.Series,
    rolling_std: pd.Series,
    alpha: float,
    n_sims: int = 10_000,
    seed: int = 42,
    chunk_size: int = 250,
) -> pd.Series:
    valid = rolling_mean.notna() & rolling_std.notna()
    valid_index = rolling_mean.index[valid]
    out = pd.Series(index=rolling_mean.index, dtype=float, name="VaR_MonteCarlo")
    rng = np.random.default_rng(seed)

    for start in range(0, len(valid_index), chunk_size):
        idx = valid_index[start : start + chunk_size]
        mu = rolling_mean.loc[idx].to_numpy()
        sigma = rolling_std.loc[idx].to_numpy()
        simulations = rng.normal(loc=mu[:, None], scale=sigma[:, None], size=(len(idx), n_sims))
        out.loc[idx] = np.quantile(simulations, alpha, axis=1)
    return out


def compute_rolling_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    window: int = 250,
    n_sims: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    confidence = normalize_confidence(confidence_level)
    alpha = 1 - confidence
    returns = pd.to_numeric(returns, errors="coerce").dropna().sort_index()

    rolling_mean = returns.rolling(window, min_periods=window).mean().shift(1)
    rolling_std = returns.rolling(window, min_periods=window).std(ddof=1).shift(1)
    historical = returns.rolling(window, min_periods=window).quantile(alpha).shift(1)
    parametric = rolling_mean + rolling_std * norm.ppf(alpha)
    monte_carlo = monte_carlo_var(rolling_mean, rolling_std, alpha, n_sims=n_sims, seed=seed)

    var_df = pd.DataFrame(
        {
            "return": returns,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "VaR_Historical": historical,
            "VaR_Normal": parametric,
            "VaR_MonteCarlo": monte_carlo,
        }
    )
    var_df.index.name = "Date"
    return add_violation_flags(var_df)


def _log_term(count: int, probability: float) -> float:
    if count == 0:
        return 0.0
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return float(count * np.log(clipped))


def kupiec_test(violation_flags: pd.Series, confidence_level: float = 0.95) -> dict:
    confidence = normalize_confidence(confidence_level)
    alpha = 1 - confidence
    flags = pd.Series(violation_flags).dropna().astype(bool)
    n_obs = int(len(flags))
    n_violations = int(flags.sum())

    if n_obs == 0:
        return {
            "observations": 0,
            "violations": 0,
            "expected_violations": np.nan,
            "violation_rate": np.nan,
            "expected_rate": alpha,
            "kupiec_lr_stat": np.nan,
            "kupiec_p_value": np.nan,
            "kupiec_pass": False,
        }

    observed_rate = n_violations / n_obs
    ll_null = _log_term(n_obs - n_violations, 1 - alpha) + _log_term(n_violations, alpha)
    ll_hat = _log_term(n_obs - n_violations, 1 - observed_rate) + _log_term(n_violations, observed_rate)
    lr_stat = max(0.0, -2 * (ll_null - ll_hat))
    p_value = float(1 - chi2.cdf(lr_stat, df=1))

    return {
        "observations": n_obs,
        "violations": n_violations,
        "expected_violations": alpha * n_obs,
        "violation_rate": observed_rate,
        "expected_rate": alpha,
        "kupiec_lr_stat": lr_stat,
        "kupiec_p_value": p_value,
        "kupiec_pass": p_value >= 0.05,
    }


def christoffersen_test(violation_flags: pd.Series, confidence_level: float = 0.95) -> dict:
    flags = pd.Series(violation_flags).dropna().astype(int)
    kupiec = kupiec_test(flags, confidence_level)

    if len(flags) < 2:
        return {
            "christoffersen_lr_stat": np.nan,
            "christoffersen_p_value": np.nan,
            "christoffersen_pass": False,
            "transition_00": 0,
            "transition_01": 0,
            "transition_10": 0,
            "transition_11": 0,
        }

    previous = flags.iloc[:-1].to_numpy()
    current = flags.iloc[1:].to_numpy()
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    total_transitions = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total_transitions if total_transitions else 0.0
    pi0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0

    restricted_ll = (
        _log_term(n00 + n10, 1 - pi)
        + _log_term(n01 + n11, pi)
    )
    unrestricted_ll = (
        _log_term(n00, 1 - pi0)
        + _log_term(n01, pi0)
        + _log_term(n10, 1 - pi1)
        + _log_term(n11, pi1)
    )
    lr_independence = max(0.0, -2 * (restricted_ll - unrestricted_ll))
    lr_conditional_coverage = kupiec["kupiec_lr_stat"] + lr_independence
    p_value = float(1 - chi2.cdf(lr_conditional_coverage, df=2))

    return {
        "christoffersen_lr_stat": lr_conditional_coverage,
        "christoffersen_p_value": p_value,
        "christoffersen_pass": p_value >= 0.05,
        "transition_00": n00,
        "transition_01": n01,
        "transition_10": n10,
        "transition_11": n11,
    }


def backtest_methods(
    var_df: pd.DataFrame,
    confidence_level: float = 0.95,
    methods: Iterable[str] = VAR_METHODS,
) -> pd.DataFrame:
    rows = []
    for method in methods:
        column = METHOD_TO_COLUMN[method]
        valid = var_df["return"].notna() & var_df[column].notna()
        flags = var_df.loc[valid, "return"].lt(var_df.loc[valid, column])
        kupiec = kupiec_test(flags, confidence_level)
        christoffersen = christoffersen_test(flags, confidence_level)
        rows.append(
            {
                "method": method,
                **kupiec,
                **christoffersen,
                "overall_pass": bool(kupiec["kupiec_pass"] and christoffersen["christoffersen_pass"]),
            }
        )
    return pd.DataFrame(rows)


def violation_counts_by_year(var_df: pd.DataFrame, methods: Iterable[str] = VAR_METHODS) -> pd.DataFrame:
    rows = []
    for method in methods:
        column = METHOD_TO_COLUMN[method]
        valid = var_df["return"].notna() & var_df[column].notna()
        flags = var_df.loc[valid, "return"].lt(var_df.loc[valid, column])
        counts = flags.groupby(flags.index.year).sum()
        for year, count in counts.items():
            rows.append({"year": int(year), "method": method, "violations": int(count)})
    return pd.DataFrame(rows, columns=["year", "method", "violations"])


def build_violation_events(
    ticker: str,
    var_df: pd.DataFrame,
    vix_series: pd.Series,
    methods: Iterable[str] = VAR_METHODS,
) -> pd.DataFrame:
    vix_series = pd.to_numeric(vix_series, errors="coerce").reindex(var_df.index).ffill().bfill()
    rows = []

    for method in methods:
        column = METHOD_TO_COLUMN[method]
        valid = var_df["return"].notna() & var_df[column].notna()
        flags = (var_df["return"].lt(var_df[column]) & valid).astype(bool)

        in_event = False
        start_date = None
        duration = 0
        severity = 0.0
        vix_at_start = np.nan

        for date, is_violation in flags.items():
            if is_violation and not in_event:
                in_event = True
                start_date = date
                duration = 1
                severity = float(var_df.at[date, column] - var_df.at[date, "return"])
                vix_at_start = float(vix_series.at[date]) if pd.notna(vix_series.at[date]) else np.nan
            elif is_violation and in_event:
                duration += 1
                severity = max(severity, float(var_df.at[date, column] - var_df.at[date, "return"]))
            elif not is_violation and in_event:
                rows.append(
                    {
                        "ticker": ticker,
                        "method": method,
                        "start_date": pd.Timestamp(start_date).date().isoformat(),
                        "duration": duration,
                        "event": 1,
                        "severity": severity,
                        "vix_at_start": vix_at_start,
                    }
                )
                in_event = False

        if in_event:
            rows.append(
                {
                    "ticker": ticker,
                    "method": method,
                    "start_date": pd.Timestamp(start_date).date().isoformat(),
                    "duration": duration,
                    "event": 0,
                    "severity": severity,
                    "vix_at_start": vix_at_start,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["ticker", "method", "start_date", "duration", "event", "severity", "vix_at_start"],
    )

