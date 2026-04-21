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
    .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; }
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 18px;
        background: #ffffff;
        min-height: 116px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
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
    h1, h2, h3 { letter-spacing: 0; }
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


try:
    returns_data, _vix_series = load_source_data()
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

st.subheader("1. Summary")
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

st.subheader("2. Main VaR Chart")
st.plotly_chart(make_var_chart(filtered, ticker, methods_for_chart, confidence_pct), width="stretch")

st.subheader("3. Violation Analysis")
violation_counts = violation_counts_by_year(filtered)
st.plotly_chart(make_violation_bar(violation_counts), width="stretch")

st.subheader("4. Backtesting Results")
backtest_table = make_backtest_table(backtests)
st.dataframe(style_backtest_table(backtest_table), width="stretch")

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
