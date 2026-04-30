import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config import (
    TICKERS,
    START_DATE,
    END_DATE,
    MOMENTUM_LOOKBACK_DAYS,
    VOL_LOOKBACK_DAYS,
    REBALANCE_FREQ,
    ANNUAL_TARGET_VOL,
    MAX_LEVERAGE,
    TRANSACTION_COST_BPS,
    WEIGHT_CHANGE_THRESHOLD,
    REGIME_FILTER_TICKER,
    REGIME_FILTER_MA_DAYS,
)
from data_api import get_prices, get_etf_news_why
from strategy import compute_target_weights
from backtest import run_backtest
from portfolios import build_portfolio
from metrics import compute_metrics, PCT_METRICS, FLOAT_METRICS, compute_undervaluation, compute_overvaluation
from etf_info import ticker_label, ETF_INFO
from alpha_strategy import run_annual_momentum
from trend_strategy import run_trend_following, CROSS_ASSET_UNIVERSE, SAFE_HAVEN_DEFAULT
from chat_agent import run_agent

st.set_page_config(page_title="Portofolio", layout="wide")
st.title("Portofolio")
st.write("Momentum + inverse-vol ETF strategy")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Tickers")

ticker_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value=", ".join(TICKERS),
    help="ETFs to include in the universe; the model will select the best ones each month.",
)
user_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
if len(user_tickers) < 1:
    st.error("Please enter at least one valid ticker.")
    st.stop()

# Fixed strategy parameters (not exposed to the user)
user_top_n          = min(10, len(user_tickers))
user_target_vol     = float(ANNUAL_TARGET_VOL)
user_use_spy_filter = False
rebalance_threshold = 0.05
weighting_method    = "Inverse Volatility"
softmax_alpha       = 4.0

# ── Chat state (initialised before download so session_state is always ready) ──
_openai_key  = st.secrets.get("OPENAI_API_KEY", "")
_chat_enabled = bool(_openai_key and not _openai_key.startswith("sk-your"))
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_chat_ts" not in st.session_state:
    st.session_state.last_chat_ts = ""

# ── Data download ─────────────────────────────────────────────────────────────
with st.spinner("Downloading data and running model..."):
    universe = sorted(set(
        user_tickers
        + CROSS_ASSET_UNIVERSE
        + [REGIME_FILTER_TICKER, "SPY", "AGG", "USO", "GLD", "HYG", "IEF", SAFE_HAVEN_DEFAULT, "^VIX"]
    ))
    try:
        prices = get_prices(universe, START_DATE, END_DATE)
    except Exception as e:
        st.error(f"Failed to download price data: {e}")
        st.stop()

    missing = [t for t in user_tickers if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        st.warning(f"No data returned for: {', '.join(missing)}. They will be ignored.")
        user_tickers = [t for t in user_tickers if t not in missing]
    if len(user_tickers) < 1:
        st.error("None of the entered tickers returned valid data.")
        st.stop()

    # Primary strategy (existing pipeline with vol scaling + TC)
    target_weights = compute_target_weights(
        prices=prices,
        tickers=user_tickers,
        momentum_lookback_days=MOMENTUM_LOOKBACK_DAYS,
        vol_lookback_days=VOL_LOOKBACK_DAYS,
        top_n=user_top_n,
        rebalance_freq=REBALANCE_FREQ,
        use_spy_filter=user_use_spy_filter,
        regime_filter_ticker=REGIME_FILTER_TICKER,
        regime_filter_ma_days=REGIME_FILTER_MA_DAYS,
        weight_change_threshold=WEIGHT_CHANGE_THRESHOLD,
        weighting_method=weighting_method,
        softmax_alpha=softmax_alpha if softmax_alpha is not None else 4.0,
    )
    results = run_backtest(
        prices=prices,
        target_weights=target_weights,
        tickers=user_tickers,
        vol_lookback_days=VOL_LOOKBACK_DAYS,
        annual_target_vol=user_target_vol,
        max_leverage=MAX_LEVERAGE,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )

    # ── Build all comparison portfolios ──────────────────────────────────────
    COMPARISON_METHODS = [
        "Inverse Volatility",
    ]

    comparison_results = {}
    for method in COMPARISON_METHODS:
        try:
            comparison_results[method] = build_portfolio(
                method=method,
                prices=prices,
                tickers=user_tickers,
                momentum_lookback=MOMENTUM_LOOKBACK_DAYS,
                vol_lookback=VOL_LOOKBACK_DAYS,
                top_n=user_top_n,
                rebalance_freq=REBALANCE_FREQ,
                use_spy_filter=user_use_spy_filter,
                regime_ticker=REGIME_FILTER_TICKER,
                regime_ma_days=REGIME_FILTER_MA_DAYS,
                weight_change_threshold=WEIGHT_CHANGE_THRESHOLD,
                softmax_alpha=softmax_alpha if softmax_alpha is not None else 4.0,
            )
        except Exception as e:
            st.warning(f"Portfolio method '{method}' failed: {e}")

    # SPY metrics
    spy_returns = prices["SPY"].pct_change().dropna()
    spy_metrics = compute_metrics(spy_returns, prices["SPY"])

    # ETF-only prices — exclude non-investable indices (e.g. ^VIX)
    prices_etf = prices[[c for c in prices.columns if not c.startswith("^")]]

    # Undervaluation scoring across full universe
    uv_df = compute_undervaluation(prices_etf)
    ov_df = compute_overvaluation(prices_etf)

    # Cross-asset trend following strategy
    _trend = run_trend_following(prices_etf, rebalance_freq="ME")
    _trend_metrics = compute_metrics(_trend["returns"], _trend["cum_returns"])

# ── Market Regime indicators ──────────────────────────────────────────────────
_vix_series = prices["^VIX"].dropna() if "^VIX" in prices.columns else pd.Series(dtype=float)
_vix_current = float(_vix_series.iloc[-1]) if not _vix_series.empty else np.nan

_spy_cummax = prices["SPY"].cummax()
_spy_dd_series = prices["SPY"] / _spy_cummax - 1.0
_spy_dd_current = float(_spy_dd_series.iloc[-1])

_hygief_vs_mean = np.nan
if "HYG" in prices.columns and "IEF" in prices.columns:
    _hygief_ratio = (prices["HYG"] / prices["IEF"]).dropna()
    _hygief_current = float(_hygief_ratio.iloc[-1])
    _hygief_ma = _hygief_ratio.rolling(252).mean().iloc[-1]
    if not np.isnan(_hygief_ma) and _hygief_ma != 0:
        _hygief_vs_mean = (_hygief_current - float(_hygief_ma)) / float(_hygief_ma)

_high_stress = (not np.isnan(_vix_current) and _vix_current > 25) or (_spy_dd_current < -0.10)
_regime_label = "HIGH STRESS" if _high_stress else "NORMAL"

# ── Derived values for Today's Recommendation ─────────────────────────────────
px = prices[user_tickers]
today = prices.index[-1]
rebalance_dates = px.resample(REBALANCE_FREQ).last().index
past_rb_dates = rebalance_dates[rebalance_dates <= today]
latest_rb = past_rb_dates[-1]
prev_rb = past_rb_dates[-2] if len(past_rb_dates) >= 2 else None

spy_series = prices[REGIME_FILTER_TICKER]
spy_ma_series = spy_series.rolling(REGIME_FILTER_MA_DAYS).mean()
spy_price_today = spy_series.iloc[-1]
spy_ma_today = spy_ma_series.iloc[-1]
spy_risk_on_today = bool(spy_price_today > spy_ma_today)

spy_price_rb = spy_series.loc[latest_rb]
spy_ma_rb = spy_ma_series.loc[latest_rb]
spy_risk_on_rb = bool(spy_price_rb > spy_ma_rb)

today_weights = target_weights.loc[latest_rb].reindex(user_tickers, fill_value=0.0)
prev_weights = (
    target_weights.loc[prev_rb].reindex(user_tickers, fill_value=0.0)
    if prev_rb is not None
    else pd.Series(0.0, index=user_tickers)
)

today_all_zero = today_weights.sum() == 0
max_weight_change = (today_weights - prev_weights).abs().max()
action = "HOLD" if max_weight_change <= rebalance_threshold else "REBALANCE"

rets_px = px.pct_change()
momentum_raw = px / px.shift(MOMENTUM_LOOKBACK_DAYS) - 1.0
vol_raw = rets_px.rolling(VOL_LOOKBACK_DAYS).std()
signal_raw = momentum_raw / vol_raw.replace(0, np.nan)
latest_signal = signal_raw.loc[latest_rb].sort_values(ascending=False)
positive_signal_tickers = latest_signal[latest_signal > 0].index.tolist()


last_data_date = prices.index[-1].date()
st.info(f"Last available market data: **{last_data_date}**")

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET REGIME
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader(
    "Market Regime",
    help=(
        "A composite stress indicator combining three signals: "
        "(1) VIX > 25 signals elevated fear, "
        "(2) SPY drawdown below −10% from its all-time high signals trend breakdown, "
        "(3) HYG/IEF ratio deviation from its 1-year mean signals credit stress. "
        "HIGH STRESS triggers a dislocation strategy that overweights undervalued ETFs."
    ),
)

_rc1, _rc2, _rc3, _rc4 = st.columns(4)

with _rc1:
    _vix_disp = f"{_vix_current:.1f}" if not np.isnan(_vix_current) else "N/A"
    _vix_delta = "above stress threshold (25)" if (not np.isnan(_vix_current) and _vix_current > 25) else "below stress threshold (25)"
    st.metric("VIX (Fear Index)", _vix_disp, delta=_vix_delta,
              delta_color="inverse" if (not np.isnan(_vix_current) and _vix_current > 25) else "off")

with _rc2:
    st.metric("SPY Drawdown", f"{_spy_dd_current:.1%}",
              delta="below −10% threshold" if _spy_dd_current < -0.10 else "within normal range",
              delta_color="inverse" if _spy_dd_current < -0.10 else "off")

with _rc3:
    if not np.isnan(_hygief_vs_mean):
        st.metric("Credit Spread Proxy (HYG/IEF)", f"{_hygief_vs_mean:+.1%} vs 1Y avg",
                  delta="spreads widening (stress)" if _hygief_vs_mean < -0.02 else "spreads normal",
                  delta_color="inverse" if _hygief_vs_mean < -0.02 else "off")
    else:
        st.metric("Credit Spread Proxy (HYG/IEF)", "N/A")

with _rc4:
    if _high_stress:
        st.error(f"**Regime: {_regime_label}**")
    else:
        st.success(f"**Regime: {_regime_label}**")

if _high_stress:
    st.warning(
        "**HIGH STRESS regime detected.** The Alpha Strategy shifts toward a dislocation strategy: "
        "momentum picks are blended with the most undervalued ETFs (50/50 blend) to capture "
        "mean-reversion potential during dislocations."
    )

# ── SPY Drawdown chart ────────────────────────────────────────────────────────
fig_dd, ax_dd = plt.subplots(figsize=(10, 3))
ax_dd.fill_between(_spy_dd_series.index, _spy_dd_series.values * 100, 0,
                   where=(_spy_dd_series.values < 0),
                   color="tomato", alpha=0.55, label="SPY Drawdown")
ax_dd.axhline(-10, color="darkred", linewidth=1.2, linestyle="--", label="−10% stress threshold")
ax_dd.axhline(0, color="black", linewidth=0.6)
ax_dd.set_ylabel("Drawdown (%)")
ax_dd.set_title("SPY Rolling Drawdown from All-Time High")
ax_dd.legend(loc="lower left", fontsize=9)
ax_dd.grid(True, alpha=0.3)

# Annotate last value
_dd_last_x = _spy_dd_series.index[-1]
_dd_last_y = _spy_dd_current * 100
ax_dd.plot(_dd_last_x, _dd_last_y, "o", color="darkred", markersize=5, zorder=5)
ax_dd.annotate(
    f"{_dd_last_y:.1f}%",
    xy=(_dd_last_x, _dd_last_y),
    xytext=(18, -18),
    textcoords="offset points",
    fontsize=9,
    color="darkred",
    arrowprops=dict(arrowstyle="-", color="darkred", lw=0.8),
)

plt.tight_layout()
st.pyplot(fig_dd)

# ── COPX Drawdown chart ───────────────────────────────────────────────────────
if "COPX" in prices.columns:
    _copx_series = prices["COPX"].dropna()
    _copx_cummax = _copx_series.cummax()
    _copx_dd_series = _copx_series / _copx_cummax - 1.0
    _copx_dd_current = float(_copx_dd_series.iloc[-1])

    fig_copx_dd, ax_copx_dd = plt.subplots(figsize=(10, 3))
    ax_copx_dd.fill_between(_copx_dd_series.index, _copx_dd_series.values * 100, 0,
                            where=(_copx_dd_series.values < 0),
                            color="peru", alpha=0.55, label="COPX Drawdown")
    ax_copx_dd.axhline(-10, color="saddlebrown", linewidth=1.2, linestyle="--", label="−10% threshold")
    ax_copx_dd.axhline(0, color="black", linewidth=0.6)
    ax_copx_dd.set_ylabel("Drawdown (%)")
    ax_copx_dd.set_title("COPX Rolling Drawdown from All-Time High")
    ax_copx_dd.legend(loc="lower left", fontsize=9)
    ax_copx_dd.grid(True, alpha=0.3)

    _copx_last_x = _copx_dd_series.index[-1]
    _copx_last_y = _copx_dd_current * 100
    ax_copx_dd.plot(_copx_last_x, _copx_last_y, "o", color="saddlebrown", markersize=5, zorder=5)
    ax_copx_dd.annotate(
        f"{_copx_last_y:.1f}%",
        xy=(_copx_last_x, _copx_last_y),
        xytext=(18, -18),
        textcoords="offset points",
        fontsize=9,
        color="saddlebrown",
        arrowprops=dict(arrowstyle="-", color="saddlebrown", lw=0.8),
    )

    plt.tight_layout()
    st.pyplot(fig_copx_dd)
else:
    st.info("COPX not in universe — add it to the ticker list to see its drawdown chart.")

# ── SLV Drawdown chart ────────────────────────────────────────────────────────
if "SLV" in prices.columns:
    _slv_series = prices["SLV"].dropna()
    _slv_cummax = _slv_series.cummax()
    _slv_dd_series = _slv_series / _slv_cummax - 1.0
    _slv_dd_current = float(_slv_dd_series.iloc[-1])

    fig_slv_dd, ax_slv_dd = plt.subplots(figsize=(10, 3))
    ax_slv_dd.fill_between(_slv_dd_series.index, _slv_dd_series.values * 100, 0,
                           where=(_slv_dd_series.values < 0),
                           color="slategray", alpha=0.55, label="SLV Drawdown")
    ax_slv_dd.axhline(-10, color="darkslategray", linewidth=1.2, linestyle="--", label="−10% threshold")
    ax_slv_dd.axhline(0, color="black", linewidth=0.6)
    ax_slv_dd.set_ylabel("Drawdown (%)")
    ax_slv_dd.set_title("SLV Rolling Drawdown from All-Time High")
    ax_slv_dd.legend(loc="lower left", fontsize=9)
    ax_slv_dd.grid(True, alpha=0.3)

    _slv_last_x = _slv_dd_series.index[-1]
    _slv_last_y = _slv_dd_current * 100
    ax_slv_dd.plot(_slv_last_x, _slv_last_y, "o", color="darkslategray", markersize=5, zorder=5)
    ax_slv_dd.annotate(
        f"{_slv_last_y:.1f}%",
        xy=(_slv_last_x, _slv_last_y),
        xytext=(18, -18),
        textcoords="offset points",
        fontsize=9,
        color="darkslategray",
        arrowprops=dict(arrowstyle="-", color="darkslategray", lw=0.8),
    )

    plt.tight_layout()
    st.pyplot(fig_slv_dd)
else:
    st.info("SLV not in universe — add it to the ticker list to see its drawdown chart.")


# ── HYG/IEF credit spread proxy chart ────────────────────────────────────────
if "HYG" in prices.columns and "IEF" in prices.columns:
    _hygief_ratio_full = (prices["HYG"] / prices["IEF"]).dropna()
    _hygief_ma_full    = _hygief_ratio_full.rolling(252).mean()

    fig_cs, ax_cs = plt.subplots(figsize=(10, 3))
    ax_cs.plot(_hygief_ratio_full.index, _hygief_ratio_full.values,
               color="steelblue", linewidth=1.5, label="HYG/IEF ratio")
    ax_cs.plot(_hygief_ma_full.index, _hygief_ma_full.values,
               color="darkorange", linewidth=1.5, linestyle="--", label="1-year MA")
    ax_cs.set_ylabel("Price Ratio")
    ax_cs.set_title("Credit Spread Proxy — HYG/IEF Ratio vs 1-Year MA")
    ax_cs.legend(fontsize=9)
    ax_cs.grid(True, alpha=0.3)
    ax_cs.annotate(
        "Ratio falling below MA → spreads widening → credit stress",
        xy=(0.01, 0.05), xycoords="axes fraction", fontsize=8, color="gray"
    )
    plt.tight_layout()
    st.pyplot(fig_cs)
else:
    st.info("HYG or IEF data unavailable — cannot plot credit spread proxy.")

# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Portfolio Comparison Table",
    help="Side-by-side metrics for SPY and each portfolio construction method, all using the same universe, date range, and SPY regime filter setting.")

# Build metrics for each comparison method
DISPLAY_NAMES = {
    "Inverse Volatility": "Inv. Vol",
}

metric_rows = {}
metric_rows["SPY"] = spy_metrics
metric_rows["Trend Follow"] = _trend_metrics

for method in COMPARISON_METHODS:
    if method not in comparison_results:
        continue
    col_name = DISPLAY_NAMES.get(method, method)
    r = comparison_results[method]["returns"]
    try:
        port_cum = comparison_results[method]["cum_returns"]
        metric_rows[col_name] = compute_metrics(r, port_cum)
    except Exception as e:
        st.warning(f"Could not compute metrics for {method}: {e}")

# Assemble summary DataFrame (metrics as rows, portfolios as columns)
all_metric_keys = list(spy_metrics.keys())
table_data = {}
for col, mets in metric_rows.items():
    table_data[col] = [mets.get(k, np.nan) for k in all_metric_keys]

summary_df = pd.DataFrame(table_data, index=all_metric_keys)

# ── Top 3 holdings rows ───────────────────────────────────────────────────────
def _top3(weights_series: pd.Series) -> list[str]:
    nz = weights_series[weights_series > 0].sort_values(ascending=False)
    result = list(nz.index[:3])
    while len(result) < 3:
        result.append("-")
    return result

top3_spy = ["SPY", "-", "-"]
top3_rows = {"SPY": top3_spy}

# Trend Follow latest weights
_trend_latest_w = _trend["weights"].iloc[-1] if not _trend["weights"].empty else pd.Series(dtype=float)
top3_rows["Trend Follow"] = _top3(_trend_latest_w)

for method in COMPARISON_METHODS:
    if method not in comparison_results:
        continue
    col_name = DISPLAY_NAMES.get(method, method)
    lw = comparison_results[method]["latest_weights"]
    top3_rows[col_name] = _top3(lw)

for i, label in enumerate(["Top Holding 1", "Top Holding 2", "Top Holding 3"]):
    row = {
        col: ticker_label(top3_rows[col][i]) if (col in top3_rows and top3_rows[col][i] != "-") else "-"
        for col in summary_df.columns
    }
    summary_df.loc[label] = pd.Series(row)

# Best Buy Candidate rows — global universe screen, same value repeated across all columns
if not uv_df.empty:
    best_ticker = uv_df.iloc[0]["Ticker"]
    best_score  = uv_df.iloc[0]["Underval Score"]
else:
    best_ticker = "—"
    best_score  = np.nan

for col in summary_df.columns:
    summary_df.loc["Best Buy Candidate", col] = ticker_label(best_ticker) if best_ticker != "—" else "—"
    summary_df.loc["Best Buy Score", col] = f"{best_score:.2f}" if not pd.isna(best_score) else "—"

# ── Format and display ────────────────────────────────────────────────────────
def _format_cell(val, metric):
    if isinstance(val, str):
        return val
    if pd.isna(val):
        return "—"
    if metric in PCT_METRICS:
        return f"{val:.2%}"
    if metric in FLOAT_METRICS:
        return f"{val:.2f}"
    return str(val)

formatted = summary_df.copy().astype(object)
for idx in summary_df.index:
    for col in summary_df.columns:
        formatted.loc[idx, col] = _format_cell(summary_df.loc[idx, col], idx)

COLUMN_HELP = {
    "SPY":          "Buy-and-hold SPY. 100% allocated to the S&P 500 at all times — used as the benchmark.",
    "Trend Follow": "Cross-asset trend following with absolute momentum. Universe: SPY, TLT, GLD, DBC, VNQ, EFA, EEM. Each month, holds only assets with a positive 12-month return (absolute momentum filter), weighted by rank. When nothing qualifies, moves 100% to IEF (short-term bonds).",
    "Inv. Vol":     "Selects the top-N ETFs by risk-adjusted momentum signal, then allocates more weight to the less volatile ones (weight ∝ 1/volatility). Steadier trends get larger positions.",
}

st.dataframe(
    formatted,
    use_container_width=True,
    column_config={
        col: st.column_config.TextColumn(col, help=COLUMN_HELP.get(col, ""))
        for col in formatted.columns
    },
)

# ── Most Undervalued ETFs table ───────────────────────────────────────────────
# ETF descriptions + why undervalued — sourced from web research (Apr 2026)
st.subheader("Most Undervalued ETFs Right Now",
    help="ETFs ranked by a composite score: 40% weight on 12-month drawdown depth, 40% on distance below 200-day MA, 20% on 1-month stabilisation/rebound. Higher score = more beaten-down but starting to stabilise.")

if not uv_df.empty:
    top5 = uv_df.head(5).copy()
    uv_news = get_etf_news_why(tuple(top5["Ticker"].tolist()))
    top5["Why"] = top5["Ticker"].map(lambda t: uv_news.get(t, "—"))
    top5["Ticker"] = top5["Ticker"].map(ticker_label)
    top5["12M Drawdown"]   = top5["12M Drawdown"].map(lambda x: f"{x:.2%}")
    top5["Dist vs 200DMA"] = top5["Dist vs 200DMA"].map(lambda x: f"{x:.2%}")
    top5["1M Return"]      = top5["1M Return"].map(lambda x: f"{x:.2%}")
    top5["Underval Score"] = top5["Underval Score"].map(lambda x: f"{x:.2f}")
    top5 = top5[["Ticker", "Why", "12M Drawdown", "Dist vs 200DMA", "1M Return", "Underval Score"]]
    st.dataframe(
        top5,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Why": st.column_config.TextColumn("Why", width="large")
        },
    )

    top5_tickers = uv_df.head(5)["Ticker"].tolist()
    fig_uv, ax_uv = plt.subplots(figsize=(10, 4))
    for ticker in top5_tickers:
        if ticker not in prices.columns:
            continue
        p = prices[ticker].dropna()
        normalized = p / p.iloc[0]
        ax_uv.plot(normalized.index, normalized.values, label=ticker, linewidth=1.5)
    ax_uv.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax_uv.set_title("Top 5 Undervalued ETFs — Normalized Price (base = 1)")
    ax_uv.legend()
    ax_uv.grid(True, alpha=0.3)
    st.pyplot(fig_uv)
else:
    st.info("Not enough price history to compute undervaluation scores.")

# ── Most Overvalued ETFs table ────────────────────────────────────────────────
st.subheader("Most Overvalued ETFs Right Now",
    help="ETFs ranked by a composite score: 40% nearness to 12-month high, 40% distance above 200-day MA, 20% 1-month return. Higher score = more stretched above trend, potentially overextended.")

if not ov_df.empty:
    top5_ov = ov_df.head(5).copy()
    ov_news = get_etf_news_why(tuple(top5_ov["Ticker"].tolist()))
    top5_ov["Why"] = top5_ov["Ticker"].map(lambda t: ov_news.get(t, "—"))
    top5_ov["Ticker"] = top5_ov["Ticker"].map(ticker_label)
    top5_ov["12M Nearness to High"] = top5_ov["12M Nearness to High"].map(lambda x: f"{x:.2%}")
    top5_ov["Dist vs 200DMA"]       = top5_ov["Dist vs 200DMA"].map(lambda x: f"{x:.2%}")
    top5_ov["1M Return"]            = top5_ov["1M Return"].map(lambda x: f"{x:.2%}")
    top5_ov["Overval Score"]        = top5_ov["Overval Score"].map(lambda x: f"{x:.2f}")
    top5_ov = top5_ov[["Ticker", "Why", "12M Nearness to High", "Dist vs 200DMA", "1M Return", "Overval Score"]]
    st.dataframe(
        top5_ov,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Why": st.column_config.TextColumn("Why", width="large")
        },
    )

    top5_ov_tickers = ov_df.head(5)["Ticker"].tolist()
    fig_ov, ax_ov = plt.subplots(figsize=(10, 4))
    for ticker in top5_ov_tickers:
        if ticker not in prices.columns:
            continue
        p = prices[ticker].dropna()
        normalized = p / p.iloc[0]
        ax_ov.plot(normalized.index, normalized.values, label=ticker, linewidth=1.5)
    ax_ov.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax_ov.set_title("Top 5 Overvalued ETFs — Normalized Price (base = 1)")
    ax_ov.legend()
    ax_ov.grid(True, alpha=0.3)
    st.pyplot(fig_ov)
else:
    st.info("Not enough price history to compute overvaluation scores.")




# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-ASSET TREND FOLLOWING
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Cross-Asset Trend Following",
    help=(
        "Universe: SPY (US equities), TLT (long bonds), GLD (gold), DBC (commodities), "
        "VNQ (real estate), EFA (developed international), EEM (emerging markets). "
        "Each month-end, only assets with a positive trailing 12-month return are held "
        "(absolute momentum filter). The rest go to IEF (short-term bonds). "
        "Qualifying assets are rank-weighted. This strategy improves Sharpe by: "
        "(1) diversifying across truly uncorrelated asset classes, and "
        "(2) stepping into bonds during broad market downtrends."
    ))
st.caption(
    "Each month, rank the cross-asset universe by trailing 12-month return. "
    "Hold only assets with **positive** momentum — the rest are replaced by IEF. "
    "Bonds and commodities zig when equities zag, cutting drawdowns without sacrificing long-run return."
)

_trend_cum = _trend["cum_returns"]
_trend_spy_rets = prices["SPY"].pct_change().dropna()
_trend_spy_cum  = (1.0 + _trend_spy_rets).cumprod()
_trend_common   = _trend_cum.index.intersection(_trend_spy_cum.index)

fig_trend, ax_trend = plt.subplots(figsize=(10, 4))
ax_trend.plot(_trend_cum.reindex(_trend_common).index,
              _trend_cum.reindex(_trend_common).values,
              label="Cross-Asset Trend Following", linewidth=2, color="mediumseagreen")
ax_trend.plot(_trend_spy_cum.reindex(_trend_common).index,
              _trend_spy_cum.reindex(_trend_common).values,
              label="SPY (buy & hold)", linewidth=1.5, color="orange", linestyle="--")
ax_trend.set_title("Cross-Asset Trend Following vs SPY — Cumulative Return (base = 1)")
ax_trend.legend()
ax_trend.grid(True, alpha=0.3)
st.pyplot(fig_trend)



# ═══════════════════════════════════════════════════════════════════════════════
# ALPHA STRATEGY — ANNUAL MOMENTUM ROTATION
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Alpha Strategy: Monthly Momentum Rotation")
st.caption(
    "Each month-end, rank every ETF in your universe by its trailing 12-month total return "
    "using only data available up to that date. Hold the **top 5** for the next month, "
    "weighted by rank (best performer gets the most weight). "
    "Always fully invested — no cash or bond fallback."
)

_alpha = run_annual_momentum(prices_etf, top_n=5, weight_method="rank", rebalance_freq="ME")
_alpha_cum = _alpha["cum_returns"]
_spy_rets  = prices["SPY"].pct_change().dropna()
_spy_cum   = (1.0 + _spy_rets).cumprod()

_common_idx = _alpha_cum.index.intersection(_spy_cum.index)
_alpha_cum  = _alpha_cum.reindex(_common_idx)
_spy_cum    = _spy_cum.reindex(_common_idx)

fig_alpha, ax_alpha = plt.subplots(figsize=(10, 4))
ax_alpha.plot(_alpha_cum.index, _alpha_cum.values, label="Monthly Momentum (top 5, rank-wt)", linewidth=2, color="steelblue")
ax_alpha.plot(_spy_cum.index,   _spy_cum.values,   label="SPY (buy & hold)",                 linewidth=1.5, color="orange", linestyle="--")
ax_alpha.set_title("Monthly Momentum Strategy vs SPY — Cumulative Return (base = 1)")
ax_alpha.legend()
ax_alpha.grid(True, alpha=0.3)
st.pyplot(fig_alpha)

# ── Regime-adjusted current allocation ───────────────────────────────────────
_alpha_log = _alpha["rebalance_log"]
if _alpha_log:
    _latest_entry = _alpha_log[-1]
    _mom_weights = pd.Series(_latest_entry["weights"])  # ticker → rank weight

    if _high_stress and not uv_df.empty:
        st.subheader("Dislocation Strategy — Regime-Adjusted Weights",
            help="Active because the Market Regime is HIGH STRESS. "
                 "Momentum weights are blended 50/50 with undervaluation scores to tilt "
                 "toward beaten-down ETFs that may benefit from mean-reversion.")
        # Build underval weights for the momentum picks (use universe overlap)
        _uv_lookup = uv_df.set_index("Ticker")["Underval Score"]
        _mom_tickers = _mom_weights.index.tolist()
        _uv_scores = _uv_lookup.reindex(_mom_tickers).fillna(0.0)
        _uv_scores = _uv_scores.clip(lower=0)
        _uv_w = _uv_scores / _uv_scores.sum() if _uv_scores.sum() > 0 else pd.Series(1.0 / len(_mom_tickers), index=_mom_tickers)
        # 50/50 blend
        _blended = 0.5 * _mom_weights + 0.5 * _uv_w
        _blended = _blended / _blended.sum()
        _blended_display = _blended.sort_values(ascending=False).rename("Blended Weight").to_frame()
        _blended_display["Momentum Wt"] = _mom_weights.reindex(_blended_display.index).map(lambda x: f"{x:.1%}" if not pd.isna(x) else "—")
        _blended_display["Underval Wt"] = _uv_w.reindex(_blended_display.index).map(lambda x: f"{x:.1%}" if not pd.isna(x) else "—")
        _blended_display["Blended Weight"] = _blended_display["Blended Weight"].map(lambda x: f"{x:.1%}")
        _blended_display.index = [ticker_label(t) for t in _blended_display.index]
        st.dataframe(_blended_display, use_container_width=True)
    else:
        _mom_label = "Normal Regime — Momentum Weights (latest month)"
        st.caption(f"**{_mom_label}**: {_latest_entry['date']}")



# ═══════════════════════════════════════════════════════════════════════════════
# TODAY'S RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════
st.header("Today's Recommendation")
st.markdown(f"**Recommendation date:** {latest_rb.date()} &nbsp;|&nbsp; **Latest price data:** {today.date()}")

if user_use_spy_filter:
    if spy_risk_on_today:
        st.success(
            f"Regime: RISK ON — SPY ({spy_price_today:.2f}) is above its "
            f"{REGIME_FILTER_MA_DAYS}-day MA ({spy_ma_today:.2f})"
        )
    else:
        st.error(
            f"Regime: RISK OFF — SPY ({spy_price_today:.2f}) is below its "
            f"{REGIME_FILTER_MA_DAYS}-day MA ({spy_ma_today:.2f})"
        )
else:
    st.info("SPY trend filter is disabled.")

if action == "REBALANCE":
    st.warning(
        f"Action: **REBALANCE** — largest weight change is "
        f"{max_weight_change:.1%}, above threshold of {rebalance_threshold:.1%}"
    )
else:
    st.success(
        f"Action: **HOLD** — largest weight change is "
        f"{max_weight_change:.1%}, within threshold of {rebalance_threshold:.1%}"
    )

if today_all_zero:
    if user_use_spy_filter and not spy_risk_on_rb:
        st.warning(
            f"Model recommends **staying in cash**. The SPY trend filter is RISK OFF "
            f"(SPY {spy_price_rb:.2f} < {REGIME_FILTER_MA_DAYS}-day MA {spy_ma_rb:.2f}). "
            "Uncheck the SPY filter in the sidebar to see signal-only weights."
        )
    elif not positive_signal_tickers:
        st.warning("Model recommends **staying in cash**. No assets had a positive signal.")
    else:
        st.warning("Model recommends **staying in cash**.")

st.subheader(
    "Target Weights: Today vs Previous Rebalance",
    help=(
        "How much capital to allocate to each ETF at the latest month-end rebalance date. "
        "Selection: top 10 ETFs ranked by 6-month momentum ÷ volatility (risk-adjusted signal). "
        "Weighting: inverse volatility — steadier ETFs receive larger positions. "
        "A REBALANCE signal means the largest weight change since the prior month exceeds 5%; "
        "HOLD means the portfolio drift is within that threshold and no trade is needed. "
        "The right column shows last month's weights for comparison."
    ),
)
col_today, col_prev = st.columns(2)
with col_today:
    st.markdown(f"**Latest Rebalance ({latest_rb.date()})**")
    today_display = today_weights[today_weights > 0].sort_values(ascending=False).rename("Weight").to_frame()
    today_display.index = [ticker_label(t) for t in today_display.index]
    st.dataframe(today_display.style.format("{:.2%}"))
with col_prev:
    label = prev_rb.date() if prev_rb is not None else "N/A"
    st.markdown(f"**Previous ({label})**")
    prev_display = prev_weights[prev_weights > 0].sort_values(ascending=False).rename("Weight").to_frame()
    prev_display.index = [ticker_label(t) for t in prev_display.index]
    st.dataframe(prev_display.style.format("{:.2%}"))

csv_df = today_weights.sort_values(ascending=False).rename("Weight").reset_index()
csv_df.columns = ["Ticker", "Weight"]
st.download_button(
    label="Download today's weights CSV",
    data=csv_df.to_csv(index=False),
    file_name="latest_target_weights.csv",
    mime="text/csv",
)




# ═══════════════════════════════════════════════════════════════════════════════
# ASK PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("💬 Ask Portofolio",
    help="Ask anything in plain English — correlations, volatility, drawdowns, "
         "news, macro context. The agent can compute statistics on your live data "
         "and search the web for current information.")

if not _chat_enabled:
    st.info("Add your OpenAI API key to `.streamlit/secrets.toml` to enable Ask Portofolio.")
else:
    for _u, _a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(_u)
        with st.chat_message("assistant"):
            st.write(_a)

    if st.session_state.chat_history:
        if st.button("Clear conversation", key="chat_clear"):
            st.session_state.chat_history = []
            st.rerun()

    if _question := st.chat_input("Ask anything about your ETFs or the market…"):
        with st.chat_message("user"):
            st.write(_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    _answer, _plots = run_agent(
                        question=_question,
                        prices=prices,
                        history=st.session_state.chat_history,
                        api_key=_openai_key,
                    )
                except Exception as _exc:
                    _answer, _plots = f"Error: {_exc}", []
            st.write(_answer)
            for _img in _plots:
                st.image(_img)
        st.session_state.chat_history.append((_question, _answer))

