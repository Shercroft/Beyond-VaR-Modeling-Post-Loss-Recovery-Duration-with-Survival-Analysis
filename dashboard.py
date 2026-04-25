from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from var_functions import (
    METHOD_TO_COLUMN,
    TICKERS,
    VAR_METHODS,
    backtest_methods,
    build_violation_events,
    compute_rolling_var,
    violation_counts_by_year,
)


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_SEED = 42
ROLLING_WINDOW = 250
N_SIMS = 10_000
TRADING_DAYS_PER_YEAR = 252
TEST_SIGNIFICANCE = 0.05

METHOD_OPTION_ALL = "All three"
METHOD_OPTIONS = [*VAR_METHODS, METHOD_OPTION_ALL]
METHOD_COLORS = {
    "Historical": "#005f73",
    "Parametric Normal": "#9b2226",
    "Monte Carlo": "#ee9b00",
}


st.set_page_config(
    page_title="VaR Backtesting Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f8fafc; }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1360px;
    }
    .metric-card {
        border: 1px solid #dbe3ea;
        border-radius: 8px;
        padding: 16px 18px;
        background: #ffffff;
        min-height: 116px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
    }
    .metric-label {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 760;
        line-height: 1.15;
        word-break: break-word;
    }
    .metric-note {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 8px;
    }
    .status-pass { border-left: 6px solid #16a34a; }
    .status-fail { border-left: 6px solid #dc2626; }
    .status-neutral { border-left: 6px solid #94a3b8; }
    .research-panel {
        border: 1px solid #dbe3ea;
        border-radius: 8px;
        padding: 18px 20px;
        background: #ffffff;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045);
        min-height: 160px;
    }
    .research-panel h4 {
        margin: 0 0 8px 0;
        color: #0f172a;
        font-size: 0.96rem;
    }
    .research-panel p {
        color: #475569;
        font-size: 0.88rem;
        line-height: 1.55;
        margin: 0;
    }
    .analysis-callout {
        border-left: 5px solid #005f73;
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
        color: #334155;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.045);
    }
    .analysis-callout strong { color: #0f172a; }
    .small-note {
        color: #64748b;
        font-size: 0.86rem;
        line-height: 1.55;
    }
    h1, h2, h3 { letter-spacing: 0; color: #0f172a; }
    div[data-testid="stSidebar"] { background: #eef3f7; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_source_data() -> tuple[pd.DataFrame, pd.Series]:
    returns_path = PROCESSED_DIR / "daily_returns.csv"
    vix_path = PROCESSED_DIR / "vix_aligned.csv"

    if not returns_path.exists() or not vix_path.exists():
        missing = [str(path) for path in [returns_path, vix_path] if not path.exists()]
        raise FileNotFoundError(f"Missing processed data file(s): {missing}")

    returns = pd.read_csv(returns_path, parse_dates=["Date"]).set_index("Date").sort_index()
    vix = pd.read_csv(vix_path, parse_dates=["Date"]).set_index("Date").sort_index()
    vix_column = "^VIX" if "^VIX" in vix.columns else vix.columns[0]
    return returns[TICKERS], pd.to_numeric(vix[vix_column], errors="coerce").rename("VIX")


@st.cache_data(show_spinner=False)
def load_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = PROCESSED_DIR / "return_summary_statistics.csv"
    normality_path = PROCESSED_DIR / "normality_tests.csv"

    summary = pd.read_csv(summary_path).set_index("ticker")
    normality = pd.read_csv(normality_path).set_index("ticker")
    return summary, normality


@st.cache_data(show_spinner="Calculating rolling VaR...")
def get_var_frame(ticker: str, confidence_pct: int) -> pd.DataFrame:
    returns, vix = load_source_data()
    var_df = compute_rolling_var(
        returns[ticker],
        confidence_level=confidence_pct / 100,
        window=ROLLING_WINDOW,
        n_sims=N_SIMS,
        seed=RANDOM_SEED,
    )
    return var_df.join(vix, how="left").ffill().bfill()


def selected_methods(method_option: str) -> list[str]:
    return VAR_METHODS if method_option == METHOD_OPTION_ALL else [method_option]


def filter_by_date(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()


def pass_label(value: bool) -> str:
    return "Passed" if bool(value) else "Failed"


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.2%}"


def format_float(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.4f}"


def render_metric_card(label: str, value: str, note: str = "", status: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="metric-card status-{status}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_methodology(confidence_pct: int) -> None:
    alpha = 1 - confidence_pct / 100
    st.markdown(
        f"""
        <div class="analysis-callout">
            <strong>Research design.</strong>
            The dashboard treats VaR as a one-day-ahead lower-tail forecast. Each estimate for day t
            uses information through t-1 only, with a {ROLLING_WINDOW}-trading-day rolling window.
            The exceedance probability is alpha = {alpha:.0%}; under correct unconditional coverage,
            the empirical violation rate should converge to {alpha:.0%}.
        </div>
        """,
        unsafe_allow_html=True,
    )

    formula_cols = st.columns(3)
    with formula_cols[0]:
        st.markdown(
            """
            <div class="research-panel">
                <h4>Returns and Historical Simulation</h4>
                <p>Returns are adjusted-close percentage returns. Historical VaR is the empirical alpha-quantile of the prior rolling window.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"r_t = \frac{P_t}{P_{t-1}} - 1")
        st.latex(r"\widehat{VaR}^{HS}_{t,\alpha}=Q_{\alpha}\left(r_{t-250},\ldots,r_{t-1}\right)")
    with formula_cols[1]:
        st.markdown(
            """
            <div class="research-panel">
                <h4>Parametric Normal VaR</h4>
                <p>The Gaussian model plugs the rolling sample mean and standard deviation into the normal lower-tail quantile.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"\widehat{VaR}^{N}_{t,\alpha}=\hat{\mu}_{t-1,250}+\hat{\sigma}_{t-1,250}\Phi^{-1}(\alpha)")
        st.latex(r"\hat{\mu}_{t-1,250}=\frac{1}{250}\sum_{i=1}^{250} r_{t-i}")
    with formula_cols[2]:
        st.markdown(
            """
            <div class="research-panel">
                <h4>Monte Carlo VaR and Violations</h4>
                <p>Monte Carlo VaR samples 10,000 normal shocks per forecast date and records the simulated alpha-quantile.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"r^{(j)}_{t}\sim\mathcal{N}\left(\hat{\mu}_{t-1,250},\hat{\sigma}_{t-1,250}^{2}\right)")
        st.latex(r"I_t=\mathbf{1}\left(r_t<\widehat{VaR}_{t,\alpha}\right)")

    with st.expander("Backtesting test statistics", expanded=False):
        st.latex(
            r"LR_{uc}=-2\log\left(\frac{(1-\alpha)^{n-x}\alpha^x}{(1-\hat{p})^{n-x}\hat{p}^x}\right)\sim\chi^2_1,\quad \hat{p}=\frac{x}{n}"
        )
        st.latex(
            r"LR_{cc}=LR_{uc}+LR_{ind}\sim\chi^2_2,\quad LR_{ind}=-2\log\left(\frac{(1-\pi)^{n_{00}+n_{10}}\pi^{n_{01}+n_{11}}}{(1-\pi_0)^{n_{00}}\pi_0^{n_{01}}(1-\pi_1)^{n_{10}}\pi_1^{n_{11}}}\right)"
        )
        st.markdown(
            """
            <p class="small-note">
            Kupiec's test evaluates unconditional coverage. Christoffersen's conditional coverage test
            adds an independence component, so clustered violations can fail even when the average
            violation rate is close to the theoretical tail probability.
            </p>
            """,
            unsafe_allow_html=True,
        )


def build_summary_values(backtests: pd.DataFrame, method_option: str, confidence_pct: int) -> dict:
    theoretical_rate = 1 - confidence_pct / 100

    if method_option == METHOD_OPTION_ALL:
        ordered = backtests.set_index("method").loc[VAR_METHODS]
        failures = int((~ordered["kupiec_pass"]).sum())
        return {
            "trading_days": f"{int(ordered['observations'].max()):,}",
            "violations": " / ".join(str(int(v)) for v in ordered["violations"]),
            "actual_rate": " / ".join(format_pct(v) for v in ordered["violation_rate"]),
            "theoretical_rate": format_pct(theoretical_rate),
            "kupiec": f"{len(VAR_METHODS) - failures}/{len(VAR_METHODS)} Passed",
            "kupiec_status": "pass" if failures == 0 else "fail",
            "note": "Historical / Parametric / Monte Carlo",
        }

    row = backtests.loc[backtests["method"].eq(method_option)].iloc[0]
    return {
        "trading_days": f"{int(row['observations']):,}",
        "violations": f"{int(row['violations']):,}",
        "actual_rate": format_pct(row["violation_rate"]),
        "theoretical_rate": format_pct(theoretical_rate),
        "kupiec": pass_label(row["kupiec_pass"]),
        "kupiec_status": "pass" if row["kupiec_pass"] else "fail",
        "note": method_option,
    }


def make_var_chart(df: pd.DataFrame, ticker: str, methods: list[str], confidence_pct: int) -> go.Figure:
    valid_df = df.loc[df[[METHOD_TO_COLUMN[m] for m in methods]].notna().any(axis=1)].copy()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.05,
        subplot_titles=(f"{ticker} Daily Returns vs {confidence_pct}% VaR", "VIX"),
    )

    fig.add_trace(
        go.Scatter(
            x=valid_df.index,
            y=valid_df["return"],
            mode="lines",
            name="Actual return",
            line=dict(color="#1f2937", width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>Return=%{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    violation_masks = []
    for method in methods:
        column = METHOD_TO_COLUMN[method]
        fig.add_trace(
            go.Scatter(
                x=valid_df.index,
                y=valid_df[column],
                mode="lines",
                name=f"{method} VaR",
                line=dict(color=METHOD_COLORS[method], width=2),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{method} VaR=%{{y:.2%}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        violation_masks.append(valid_df["return"].lt(valid_df[column]) & valid_df[column].notna())

    if violation_masks:
        any_violation = np.logical_or.reduce([mask.to_numpy() for mask in violation_masks])
    else:
        any_violation = np.zeros(len(valid_df), dtype=bool)

    violation_df = valid_df.loc[any_violation]
    fig.add_trace(
        go.Scatter(
            x=violation_df.index,
            y=violation_df["return"],
            mode="markers",
            name="Violation",
            marker=dict(color="#dc2626", size=7, line=dict(width=0)),
            hovertemplate="%{x|%Y-%m-%d}<br>Return=%{y:.2%}<extra>Violation</extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_hline(y=0, line_width=1, line_color="#94a3b8", opacity=0.45, row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=valid_df.index,
            y=valid_df["VIX"],
            mode="lines",
            name="VIX",
            line=dict(color="#475569", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(148, 163, 184, 0.25)",
            hovertemplate="%{x|%Y-%m-%d}<br>VIX=%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=650,
        margin=dict(l=20, r=20, t=70, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_yaxes(title_text="Daily return", tickformat=".1%", row=1, col=1)
    fig.update_yaxes(title_text="VIX", row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.25)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.25)")
    return fig


def make_violation_bar(counts: pd.DataFrame) -> go.Figure:
    if counts.empty:
        return go.Figure()

    fig = px.bar(
        counts,
        x="year",
        y="violations",
        color="method",
        barmode="group",
        color_discrete_map=METHOD_COLORS,
        labels={"year": "Year", "violations": "Violations", "method": "VaR method"},
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(dtick=1, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.25)")
    return fig


def make_backtest_table(backtests: pd.DataFrame) -> pd.DataFrame:
    indexed = backtests.set_index("method").loc[VAR_METHODS]
    table = pd.DataFrame(
        {
            method: {
                "Observations": f"{int(row['observations']):,}",
                "Violations": f"{int(row['violations']):,}",
                "Violation rate": format_pct(row["violation_rate"]),
                "Kupiec LR": format_float(row["kupiec_lr_stat"]),
                "Kupiec p-value": format_float(row["kupiec_p_value"]),
                "Kupiec result": pass_label(row["kupiec_pass"]),
                "Christoffersen LR": format_float(row["christoffersen_lr_stat"]),
                "Christoffersen p-value": format_float(row["christoffersen_p_value"]),
                "Christoffersen result": pass_label(row["christoffersen_pass"]),
                "Overall result": pass_label(row["overall_pass"]),
            }
            for method, row in indexed.iterrows()
        }
    )
    return table


def style_backtest_table(table: pd.DataFrame):
    def color_status(value):
        if value == "Passed":
            return "color: #15803d; font-weight: 700;"
        if value == "Failed":
            return "color: #b91c1c; font-weight: 700;"
        return ""

    return table.style.map(color_status)


def make_distribution_table(
    ticker: str,
    summary_stats: pd.DataFrame,
    normality_tests: pd.DataFrame,
) -> pd.DataFrame:
    summary_row = summary_stats.loc[ticker]
    normality_row = normality_tests.loc[ticker]

    rows = {
        "Daily mean": format_pct(summary_row["mean"]),
        "Annualized mean": format_pct(summary_row["mean"] * TRADING_DAYS_PER_YEAR),
        "Daily volatility": format_pct(summary_row["std"]),
        "Annualized volatility": format_pct(summary_row["std"] * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "Skewness": format_float(summary_row["skew"]),
        "Excess kurtosis": format_float(summary_row["excess_kurtosis"]),
        "Jarque-Bera p-value": format_float(normality_row["jarque_bera_p_value"]),
        "D'Agostino K2 p-value": format_float(normality_row["dagostino_k2_p_value"]),
        "Normality at 5%": "Rejected"
        if bool(normality_row["reject_normality_5pct_jb"] or normality_row["reject_normality_5pct_k2"])
        else "Not rejected",
    }
    return pd.DataFrame({"Statistic": rows.keys(), "Value": rows.values()})


def make_method_diagnostics(
    df: pd.DataFrame,
    backtests: pd.DataFrame,
    methods: list[str],
) -> pd.DataFrame:
    rows = []
    indexed_backtests = backtests.set_index("method")

    for method in methods:
        column = METHOD_TO_COLUMN[method]
        valid = df["return"].notna() & df[column].notna()
        flags = df.loc[valid, "return"].lt(df.loc[valid, column])
        severity = (df.loc[valid, column] - df.loc[valid, "return"]).where(flags)
        vix_valid = df.loc[valid, "VIX"]
        transition_denominator = indexed_backtests.at[method, "transition_10"] + indexed_backtests.at[method, "transition_11"]
        hit_after_hit = (
            indexed_backtests.at[method, "transition_11"] / transition_denominator
            if transition_denominator
            else np.nan
        )

        rows.append(
            {
                "Method": method,
                "Violations": int(flags.sum()),
                "Violation rate": format_pct(flags.mean()),
                "Mean severity": format_pct(severity.mean()),
                "Max severity": format_pct(severity.max()),
                "Mean VIX on violation": format_float(vix_valid.loc[flags].mean()),
                "Mean VIX otherwise": format_float(vix_valid.loc[~flags].mean()),
                "Hit-after-hit probability": format_pct(hit_after_hit),
            }
        )

    return pd.DataFrame(rows)


def render_empirical_interpretation(
    ticker: str,
    methods: list[str],
    backtests: pd.DataFrame,
    summary_stats: pd.DataFrame,
    normality_tests: pd.DataFrame,
    confidence_pct: int,
) -> None:
    summary_row = summary_stats.loc[ticker]
    normality_row = normality_tests.loc[ticker]
    indexed = backtests.set_index("method").loc[methods]

    kurtosis = float(summary_row["excess_kurtosis"])
    skew = float(summary_row["skew"])
    avg_rate = float(indexed["violation_rate"].mean())
    expected_rate = 1 - confidence_pct / 100
    kupiec_passes = int(indexed["kupiec_pass"].sum())
    christoffersen_passes = int(indexed["christoffersen_pass"].sum())

    skew_text = "left-skewed downside tail" if skew < -0.05 else "near-symmetric return distribution"
    kurtosis_text = "strong fat-tail behavior" if kurtosis > 3 else "moderate tail thickness"
    normality_text = (
        "normality is rejected at the 5% level, supporting non-Gaussian tail diagnostics"
        if bool(normality_row["reject_normality_5pct_jb"] or normality_row["reject_normality_5pct_k2"])
        else "normality is not rejected at the 5% level in these diagnostics"
    )
    coverage_text = (
        "close to"
        if abs(avg_rate - expected_rate) <= 0.005
        else "above"
        if avg_rate > expected_rate
        else "below"
    )

    st.markdown(
        f"""
        <div class="analysis-callout">
            <strong>{ticker} empirical reading.</strong>
            The selected sample shows {skew_text} and {kurtosis_text}
            (skewness {skew:.3f}, excess kurtosis {kurtosis:.3f}); {normality_text}.
            Across the selected method set, the average violation rate is {avg_rate:.2%},
            {coverage_text} the theoretical {expected_rate:.2%} level. Kupiec coverage passes
            for {kupiec_passes}/{len(methods)} method(s), while Christoffersen conditional coverage
            passes for {christoffersen_passes}/{len(methods)} method(s), highlighting whether
            exceptions arrive independently or in volatility clusters.
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    returns_data, _vix_series = load_source_data()
    summary_stats, normality_tests = load_diagnostics()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

min_date = returns_data.index.min().date()
max_date = returns_data.index.max().date()

st.sidebar.title("Controls")
ticker = st.sidebar.selectbox("Ticker", TICKERS, index=0)
method_option = st.sidebar.selectbox("VaR method", METHOD_OPTIONS, index=3)
confidence_pct = st.sidebar.select_slider("Confidence level", options=[90, 95, 99], value=95)
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

methods_for_chart = selected_methods(method_option)
var_frame = get_var_frame(ticker, confidence_pct)
filtered = filter_by_date(var_frame, start_date, end_date)

if filtered.empty:
    st.warning("No data is available for the selected date range.")
    st.stop()

backtests = backtest_methods(filtered, confidence_level=confidence_pct / 100)
summary = build_summary_values(backtests, method_option, confidence_pct)

st.title("VaR Backtesting Dashboard")
st.caption(
    f"{ticker} | {confidence_pct}% confidence | {ROLLING_WINDOW}-day rolling window | "
    f"{N_SIMS:,} Monte Carlo paths | {start_date} to {end_date}"
)

st.subheader("1. Methodology and Formulae")
render_methodology(confidence_pct)

st.subheader("2. Summary")
metric_cols = st.columns(5)
with metric_cols[0]:
    render_metric_card("Total trading days", summary["trading_days"], summary["note"])
with metric_cols[1]:
    render_metric_card("Violations", summary["violations"], summary["note"])
with metric_cols[2]:
    render_metric_card("Actual violation rate", summary["actual_rate"], summary["note"])
with metric_cols[3]:
    render_metric_card("Theoretical violation rate", summary["theoretical_rate"], f"{confidence_pct}% VaR")
with metric_cols[4]:
    render_metric_card("Kupiec Test", summary["kupiec"], "5% test threshold", summary["kupiec_status"])

st.subheader("3. Main VaR Chart")
st.plotly_chart(make_var_chart(filtered, ticker, methods_for_chart, confidence_pct), width="stretch")

st.subheader("4. Violation Analysis")
violation_counts = violation_counts_by_year(filtered)
st.plotly_chart(make_violation_bar(violation_counts), width="stretch")

st.subheader("5. Backtesting Results")
backtest_table = make_backtest_table(backtests)
st.dataframe(style_backtest_table(backtest_table), width="stretch")

st.subheader("6. Empirical Diagnostics and Interpretation")
render_empirical_interpretation(
    ticker,
    methods_for_chart,
    backtests,
    summary_stats,
    normality_tests,
    confidence_pct,
)
diagnostic_cols = st.columns([0.82, 1.18])
with diagnostic_cols[0]:
    st.markdown("**Return distribution diagnostics**")
    st.dataframe(make_distribution_table(ticker, summary_stats, normality_tests), width="stretch", hide_index=True)
with diagnostic_cols[1]:
    st.markdown("**Violation severity and VIX context**")
    st.dataframe(make_method_diagnostics(filtered, backtests, methods_for_chart), width="stretch", hide_index=True)

st.subheader("Survival Analysis Export")
events = build_violation_events(ticker, filtered, filtered["VIX"], methods=methods_for_chart)
csv_payload = events.to_csv(index=False).encode("utf-8")

download_cols = st.columns([1, 2])
with download_cols[0]:
    st.download_button(
        "Download violation events CSV",
        data=csv_payload,
        file_name=f"{ticker}_{confidence_pct}_var_violation_events.csv",
        mime="text/csv",
        disabled=events.empty,
    )
with download_cols[1]:
    st.caption(
        "Export columns: ticker, method, start_date, duration, event, severity, vix_at_start. "
        "For All three, the file includes all three VaR methods."
    )

if events.empty:
    st.info("No violation events are available for the current filters.")
else:
    st.dataframe(events.head(20), width="stretch")
