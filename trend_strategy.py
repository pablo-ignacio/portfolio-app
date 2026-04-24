# trend_strategy.py — Cross-Asset Trend Following with Absolute Momentum
#
# Strategy logic
# --------------
# At each month-end rebalance date, using only data available up to that point:
#   1. Compute trailing 12-month total return for each asset in the cross-asset universe.
#   2. ABSOLUTE MOMENTUM FILTER: keep only assets with a positive 12M return.
#      Assets in a downtrend are excluded regardless of relative ranking.
#   3. RANK WEIGHT the qualifying assets (best momentum → largest weight).
#   4. If NO assets qualify (every asset in a downtrend), allocate 100% to the
#      safe-haven bond ETF (IEF). This is the key difference from purely
#      relative momentum strategies — it steps aside during broad bear markets.
#
# Why this improves Sharpe vs SPY
# --------------------------------
# The cross-asset universe (equities + bonds + commodities + real estate) is
# genuinely diversified: bonds and gold tend to rise when equities fall.
# The absolute momentum filter dramatically cuts drawdowns by exiting into bonds
# during sustained downtrends — the single biggest lever on Sharpe ratio.

import numpy as np
import pandas as pd

# Default cross-asset universe — covers all major risk regimes
CROSS_ASSET_UNIVERSE = ["SPY", "TLT", "GLD", "DBC", "VNQ", "EFA", "EEM"]
SAFE_HAVEN_DEFAULT   = "IEF"


def run_trend_following(
    prices: pd.DataFrame,
    lookback_days: int = 252,    # trailing window for absolute momentum signal (~1 year)
    rebalance_freq: str = "ME",  # month-end rebalance
    safe_haven: str = SAFE_HAVEN_DEFAULT,
    cross_asset_tickers: list | None = None,
) -> dict:
    """
    Cross-asset trend following with absolute momentum.

    Parameters
    ----------
    prices             : DataFrame of adjusted close prices (dates × tickers)
    lookback_days      : trailing days used to compute momentum (~252 = 1 year)
    rebalance_freq     : pandas offset alias for rebalance dates (default "ME")
    safe_haven         : ticker to hold when no assets have positive momentum
    cross_asset_tickers: override the default cross-asset universe

    Returns
    -------
    dict with:
      "weights"          : pd.DataFrame  rebalance-date weights
      "returns"          : pd.Series     daily portfolio returns
      "cum_returns"      : pd.Series     cumulative return (starting at 1.0)
      "ticker_counts"    : pd.Series     how often each ticker was held
      "rebalance_log"    : list of dicts, one entry per rebalance
      "available_tickers": list of tickers from the universe that had price data
    """
    if cross_asset_tickers is None:
        cross_asset_tickers = CROSS_ASSET_UNIVERSE

    # Only work with tickers that actually have price data
    available = [
        t for t in cross_asset_tickers
        if t in prices.columns and not prices[t].dropna().empty
    ]

    rets = prices.pct_change()
    rebalance_dates = prices.resample(rebalance_freq).last().index
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    rebal_log = []

    for dt in rebalance_dates:
        hist = prices.loc[:dt]
        if len(hist) < lookback_days + 5:
            continue

        # Trailing 12M return — information available at this date
        mom = hist[available].iloc[-1] / hist[available].iloc[-(lookback_days + 1)] - 1.0
        mom = mom.dropna()

        # Absolute momentum filter: keep only assets trending up
        qualifying = mom[mom > 0].sort_values(ascending=False)

        if len(qualifying) == 0:
            # Broad downtrend — park in safe-haven bonds
            if safe_haven in prices.columns:
                weights.loc[dt, safe_haven] = 1.0
                rebal_log.append({
                    "date":    dt.date(),
                    "picks":   [safe_haven],
                    "weights": {safe_haven: 1.0},
                    "12M ret": {safe_haven: float(mom.get(safe_haven, np.nan))},
                    "in_safe_haven": True,
                })
            continue

        # Rank weight among qualifying assets (best momentum → most weight)
        n = len(qualifying)
        rank_vals = np.arange(n, 0, -1, dtype=float)
        w = pd.Series(rank_vals / rank_vals.sum(), index=qualifying.index)

        weights.loc[dt, w.index] = w.values
        rebal_log.append({
            "date":    dt.date(),
            "picks":   qualifying.index.tolist(),
            "weights": w.round(3).to_dict(),
            "12M ret": qualifying.round(3).to_dict(),
            "in_safe_haven": False,
        })

    # Daily portfolio returns — weights held from day after rebalance
    weights_daily = weights.reindex(rets.index).ffill().fillna(0.0).shift(1).fillna(0.0)
    common_cols   = [c for c in weights_daily.columns if c in rets.columns]
    port_returns  = (weights_daily[common_cols] * rets[common_cols]).sum(axis=1).dropna()
    cum_returns   = (1.0 + port_returns).cumprod()

    ticker_counts = (weights > 0.01).sum().sort_values(ascending=False)
    ticker_counts = ticker_counts[ticker_counts > 0]

    return {
        "weights":           weights,
        "returns":           port_returns,
        "cum_returns":       cum_returns,
        "ticker_counts":     ticker_counts,
        "rebalance_log":     rebal_log,
        "available_tickers": available,
    }
