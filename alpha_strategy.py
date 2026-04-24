# alpha_strategy.py — Annual Momentum Strategy
#
# Design rationale
# ----------------
# The previous dual-momentum approach was too conservative: the absolute-momentum
# and trend filters pushed the portfolio into bonds (AGG) too frequently, which
# crushed returns relative to SPY.
#
# This strategy takes the opposite tack:
#   - Stay ALWAYS invested — no defensive/cash mode.
#   - Rebalance ONCE per year (year-end) so we aren't trading noise.
#   - At each rebalance date use ONLY data available up to that point.
#   - Pick the top N ETFs by trailing 12-month total return (cross-sectional
#     momentum — last year's winners tend to lead next year too).
#   - Weight by RANK: best ETF gets proportionally the most weight.
#
# With a broad universe that includes sector ETFs (QQQ, XLK, XLE, XLV…),
# commodities (GLD, DBC…) and international equities, annual momentum rotation
# captures whichever asset class is leading without timing the market.

import numpy as np
import pandas as pd


def run_annual_momentum(
    prices: pd.DataFrame,
    top_n: int = 5,
    lookback_days: int = 252,   # trailing window for ranking (~1 year)
    vol_days: int = 60,          # vol window used only for inv-vol tiebreak
    rebalance_freq: str = "ME",  # month-end rebalance
    weight_method: str = "rank", # "rank" | "equal" | "inv_vol"
) -> dict:
    """
    Annual cross-sectional momentum strategy.

    Each year-end rebalance date (using only data up to that date):
      1. Compute the trailing `lookback_days` return for every ticker.
      2. Sort descending; select the top `top_n`.
      3. Assign weights by rank (top rank → largest weight), equal, or inv-vol.
      4. Hold the portfolio unchanged until the next year-end.

    Parameters
    ----------
    prices         : DataFrame of adjusted close prices (dates × tickers)
    top_n          : number of ETFs to hold each year
    lookback_days  : trailing days used to rank ETFs (~252 = 1 year)
    vol_days       : rolling vol window (used only when weight_method="inv_vol")
    rebalance_freq : pandas offset alias for rebalance dates (default "ME")
    weight_method  : how to distribute weight among chosen ETFs
                     "rank"    → proportional to rank position (best = highest)
                     "equal"   → 1/N
                     "inv_vol" → inverse trailing volatility

    Returns
    -------
    dict with:
      "weights"       : pd.DataFrame  rebalance-date weights (dates × tickers)
      "returns"       : pd.Series     daily portfolio returns
      "cum_returns"   : pd.Series     cumulative return (starting at 1.0)
      "ticker_counts" : pd.Series     # rebalances each ticker was held (sorted desc)
      "rebalance_log" : list of dicts, one entry per rebalance with date + picks
    """
    rets = prices.pct_change()
    vol  = rets.rolling(vol_days).std()

    rebalance_dates = prices.resample(rebalance_freq).last().index

    weights     = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    rebal_log   = []

    for dt in rebalance_dates:
        hist_prices = prices.loc[:dt]
        if len(hist_prices) < lookback_days + 5:
            continue  # not enough history yet

        # Trailing 12M return — information available at this date
        mom = hist_prices.iloc[-1] / hist_prices.iloc[-(lookback_days + 1)] - 1.0
        mom = mom.dropna()

        top = mom.sort_values(ascending=False).head(top_n)
        chosen = top.index.tolist()
        n = len(chosen)
        if n == 0:
            continue

        if weight_method == "equal":
            w = pd.Series(1.0 / n, index=chosen)

        elif weight_method == "inv_vol":
            hist_vol = vol.loc[:dt].iloc[-1]
            iv = 1.0 / hist_vol.loc[chosen].replace(0, np.nan)
            iv = iv.dropna()
            if iv.sum() == 0:
                w = pd.Series(1.0 / n, index=chosen)
            else:
                w = iv / iv.sum()

        else:  # rank weight (default)
            rank_vals = np.arange(n, 0, -1, dtype=float)  # n … 1
            w = pd.Series(rank_vals / rank_vals.sum(), index=chosen)

        weights.loc[dt, w.index] = w.values

        rebal_log.append({
            "date":    dt.date(),
            "picks":   chosen,
            "weights": w.round(3).to_dict(),
            "12M ret": top.round(3).to_dict(),
        })

    # Daily portfolio returns — weights held from day after rebalance
    weights_daily = weights.reindex(rets.index).ffill().fillna(0.0).shift(1).fillna(0.0)
    port_returns  = (weights_daily * rets).sum(axis=1).dropna()
    cum_returns   = (1.0 + port_returns).cumprod()

    ticker_counts = (weights > 0.01).sum().sort_values(ascending=False)
    ticker_counts = ticker_counts[ticker_counts > 0]

    return {
        "weights":       weights,
        "returns":       port_returns,
        "cum_returns":   cum_returns,
        "ticker_counts": ticker_counts,
        "rebalance_log": rebal_log,
    }
