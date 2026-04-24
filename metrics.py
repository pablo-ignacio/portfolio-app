# metrics.py — shared metric computation for all portfolios and benchmarks

import numpy as np
import pandas as pd


def _momentum(prices: pd.Series, days: int) -> float:
    """Total return over trailing `days` trading days."""
    valid = prices.dropna()
    if len(valid) <= days:
        return np.nan
    return float(valid.iloc[-1] / valid.iloc[-(days + 1)] - 1.0)


def compute_metrics(returns: pd.Series, prices: pd.Series | None = None) -> dict:
    """
    Compute a standard set of metrics from a daily return series.

    Parameters
    ----------
    returns : daily return series (net)
    prices  : optional price series used for momentum lookbacks
    """
    r = returns.dropna()
    if len(r) < 2:
        return {k: np.nan for k in _metric_keys()}

    cum = (1.0 + r).cumprod()
    n = len(r)

    ann_return = (1.0 + r).prod() ** (252 / n) - 1.0
    daily_mean = float(r.mean())
    daily_vol = float(r.std())
    ann_vol = daily_vol * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    downside = r[r < 0].std() * np.sqrt(252)
    sortino = ann_return / downside if downside > 0 else np.nan

    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    mdd = float(drawdown.min())
    calmar = ann_return / abs(mdd) if mdd != 0 else np.nan

    hit_rate = float((r > 0).mean())

    # Momentum from price series (or reconstruct from returns if prices not supplied)
    if prices is not None:
        p = prices.reindex(r.index).ffill()
    else:
        p = cum  # use the cumulative return as a proxy price

    mom_1m = _momentum(p, 21)
    mom_3m = _momentum(p, 63)
    mom_6m = _momentum(p, 126)
    mom_12m = _momentum(p, 252)

    return {
        "Avg Daily Return": daily_mean,
        "Ann. Return": ann_return,
        "Daily Vol": daily_vol,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": mdd,
        "Calmar": calmar,
        "1M Momentum": mom_1m,
        "3M Momentum": mom_3m,
        "6M Momentum": mom_6m,
        "12M Momentum": mom_12m,
        "Hit Rate": hit_rate,
    }


def _metric_keys() -> list[str]:
    return [
        "Avg Daily Return", "Ann. Return", "Daily Vol", "Ann. Vol",
        "Sharpe", "Sortino", "Max Drawdown", "Calmar",
        "1M Momentum", "3M Momentum", "6M Momentum", "12M Momentum",
        "Hit Rate",
    ]


# Metrics that should be displayed as percentages
PCT_METRICS = {
    "Avg Daily Return", "Ann. Return", "Daily Vol", "Ann. Vol",
    "Max Drawdown", "1M Momentum", "3M Momentum", "6M Momentum",
    "12M Momentum", "Hit Rate",
}

# Metrics displayed as 2-decimal floats
FLOAT_METRICS = {"Sharpe", "Sortino", "Calmar"}


# ── Undervaluation scoring ────────────────────────────────────────────────────

def compute_undervaluation(prices: pd.DataFrame, min_history: int = 252) -> pd.DataFrame:
    """
    Score each ETF on how attractive it looks as a buy candidate right now.

    Three components, all measured at the latest available date:

    1. 12-Month Drawdown (weight 40%)
       current price / rolling-252-day-max - 1
       More negative = more beaten down from recent highs = more attractive.
       We NEGATE before scoring so a bigger drawdown gives a higher score.

    2. Distance Below 200-Day MA (weight 40%)
       current price / 200-day SMA - 1
       More negative = further below its long-run trend = cheaper relative to trend.
       We NEGATE before scoring.

    3. Short-Term Reversal / Stabilisation (weight 20%)
       1-month return (21 trading days).
       Logic: we want ETFs that were beaten down AND have started recovering or
       at least stopped falling. A slightly positive or near-zero 1-month return
       after a large drawdown is the ideal pattern (stabilisation before rebound).
       We use the raw 1-month return directly: a small positive is best, a
       large positive means it already recovered, a large negative means still
       in freefall. This term is scored AS-IS (no negation) because a higher
       recent return means better stabilisation/recovery.

    Final score = 0.4 * z(neg_drawdown) + 0.4 * z(neg_dist_200ma) + 0.2 * z(mom_1m)

    where z() is a cross-sectional z-score across all ETFs on the latest date.
    Higher score = more attractive buy candidate.

    Returns
    -------
    DataFrame with columns:
        Ticker, 12M Drawdown, Dist vs 200DMA, 1M Return, Underval Score
    sorted by score descending.
    """
    records = []
    latest = prices.index[-1]

    for ticker in prices.columns:
        p = prices[ticker].dropna()
        if len(p) < min_history:
            continue  # not enough history to score reliably

        current = float(p.iloc[-1])

        # 1. 12-month drawdown
        roll_max_252 = p.rolling(252).max()
        if roll_max_252.iloc[-1] == 0 or pd.isna(roll_max_252.iloc[-1]):
            continue
        drawdown_12m = current / float(roll_max_252.iloc[-1]) - 1.0

        # 2. Distance below 200-day MA
        ma200 = p.rolling(200).mean()
        if pd.isna(ma200.iloc[-1]) or ma200.iloc[-1] == 0:
            continue
        dist_200ma = current / float(ma200.iloc[-1]) - 1.0

        # 3. 1-month return (21-day)
        if len(p) < 22:
            continue
        mom_1m = current / float(p.iloc[-22]) - 1.0

        records.append({
            "Ticker": ticker,
            "12M Drawdown": drawdown_12m,
            "Dist vs 200DMA": dist_200ma,
            "1M Return": mom_1m,
        })

    if not records:
        return pd.DataFrame(columns=["Ticker", "12M Drawdown", "Dist vs 200DMA", "1M Return", "Underval Score"])

    df = pd.DataFrame(records).set_index("Ticker")

    # Cross-sectional z-scores (mean=0, std=1 across all ETFs)
    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    # Negate drawdown and dist_200ma: more negative raw value → higher z-score after negation
    z_drawdown = _zscore(-df["12M Drawdown"])
    z_dist     = _zscore(-df["Dist vs 200DMA"])
    z_reversal = _zscore(df["1M Return"])

    df["Underval Score"] = 0.4 * z_drawdown + 0.4 * z_dist + 0.2 * z_reversal

    return (
        df.reset_index()
        .sort_values("Underval Score", ascending=False)
        .reset_index(drop=True)
    )


def compute_overvaluation(prices: pd.DataFrame, min_history: int = 252) -> pd.DataFrame:
    """
    Score each ETF on how stretched or overvalued it looks right now.

    Three components, all measured at the latest available date:

    1. 12-Month Nearness to High (weight 40%)
       current price / rolling-252-day-max - 1
       More positive (near or at 12-month high) = more extended = more overvalued.
       Used AS-IS before z-scoring (opposite of undervaluation).

    2. Distance Above 200-Day MA (weight 40%)
       current price / 200-day SMA - 1
       More positive = further above long-run trend = more stretched.
       Used AS-IS.

    3. Short-Term Momentum Stretch (weight 20%)
       1-month return.
       A large recent gain signals the ETF may be overbought in the short term.
       Used AS-IS (large positive = more overvalued).

    Final score = 0.4 * z(nearness_to_high) + 0.4 * z(dist_200ma) + 0.2 * z(mom_1m)
    Higher score = more stretched / potentially overvalued.
    """
    records = []

    for ticker in prices.columns:
        p = prices[ticker].dropna()
        if len(p) < min_history:
            continue

        current = float(p.iloc[-1])

        roll_max_252 = p.rolling(252).max()
        if pd.isna(roll_max_252.iloc[-1]) or roll_max_252.iloc[-1] == 0:
            continue
        nearness_to_high = current / float(roll_max_252.iloc[-1]) - 1.0

        ma200 = p.rolling(200).mean()
        if pd.isna(ma200.iloc[-1]) or ma200.iloc[-1] == 0:
            continue
        dist_200ma = current / float(ma200.iloc[-1]) - 1.0

        if len(p) < 22:
            continue
        mom_1m = current / float(p.iloc[-22]) - 1.0

        records.append({
            "Ticker": ticker,
            "12M Nearness to High": nearness_to_high,
            "Dist vs 200DMA": dist_200ma,
            "1M Return": mom_1m,
        })

    if not records:
        return pd.DataFrame(columns=["Ticker", "12M Nearness to High", "Dist vs 200DMA", "1M Return", "Overval Score"])

    df = pd.DataFrame(records).set_index("Ticker")

    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    z_nearness = _zscore(df["12M Nearness to High"])
    z_dist     = _zscore(df["Dist vs 200DMA"])
    z_reversal = _zscore(df["1M Return"])

    df["Overval Score"] = 0.4 * z_nearness + 0.4 * z_dist + 0.2 * z_reversal

    return (
        df.reset_index()
        .sort_values("Overval Score", ascending=False)
        .reset_index(drop=True)
    )


def compute_peace_dividend(
    prices: pd.DataFrame,
    min_history: int = 252,
    oil_ticker: str = "USO",
    gold_ticker: str = "GLD",
) -> pd.DataFrame:
    """
    Peace Dividend Score — ranks ETFs by how much they should rally if
    oil prices drop sharply and geopolitical fear unwinds (e.g. a war ending).

    Three components:

    1. Oil Beta (weight 50%)
       OLS slope of ETF returns on USO returns over trailing 252 days.
       Negated before scoring: a strongly negative beta (e.g. oil importers like
       Japan, India, China) scores highest because they benefit most from cheaper oil.

    2. Distance Below 200-Day MA (weight 30%)
       current price / 200-day SMA − 1.
       ETFs trading well below trend likely carry a "war risk discount" that
       unwinds when peace comes. Negated before scoring.

    3. Gold Correlation (weight 20%)
       Pearson correlation of ETF returns with GLD returns over trailing 252 days.
       Negated before scoring: ETFs that move opposite to gold (risk-on assets)
       rally as safe-haven demand falls.

    Final score = 0.5·z(−oil_beta) + 0.3·z(−dist_200ma) + 0.2·z(−gold_corr)
    Higher score = larger expected rally if the war ends / oil drops.
    """
    rets = prices.pct_change().dropna(how="all")

    if oil_ticker not in prices.columns or gold_ticker not in prices.columns:
        return pd.DataFrame(
            columns=["Ticker", "Oil Beta", "Dist vs 200DMA", "Gold Corr", "Peace Dividend Score"]
        )

    oil_rets  = rets[oil_ticker].dropna()
    gold_rets = rets[gold_ticker].dropna()

    records = []
    for ticker in prices.columns:
        if ticker in (oil_ticker, gold_ticker):
            continue

        p = prices[ticker].dropna()
        if len(p) < min_history:
            continue

        r = rets[ticker].dropna()
        common = r.index.intersection(oil_rets.index).intersection(gold_rets.index)
        if len(common) < min_history:
            continue

        r_w    = r.loc[common].iloc[-min_history:]
        oil_w  = oil_rets.loc[common].iloc[-min_history:]
        gold_w = gold_rets.loc[common].iloc[-min_history:]

        # 1. Oil beta (OLS slope)
        oil_var  = float(oil_w.var())
        oil_beta = float(r_w.cov(oil_w) / oil_var) if oil_var > 0 else 0.0

        # 2. Distance vs 200-day MA
        ma200 = p.rolling(200).mean()
        if pd.isna(ma200.iloc[-1]) or ma200.iloc[-1] == 0:
            continue
        dist_200ma = float(p.iloc[-1] / ma200.iloc[-1] - 1.0)

        # 3. Gold correlation
        gold_corr = float(r_w.corr(gold_w))
        if pd.isna(gold_corr):
            gold_corr = 0.0

        records.append({
            "Ticker":         ticker,
            "Oil Beta":       oil_beta,
            "Dist vs 200DMA": dist_200ma,
            "Gold Corr":      gold_corr,
        })

    if not records:
        return pd.DataFrame(
            columns=["Ticker", "Oil Beta", "Dist vs 200DMA", "Gold Corr", "Peace Dividend Score"]
        )

    df = pd.DataFrame(records).set_index("Ticker")

    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    z_oil  = _zscore(-df["Oil Beta"])
    z_dist = _zscore(-df["Dist vs 200DMA"])
    z_gold = _zscore(-df["Gold Corr"])

    df["Peace Dividend Score"] = 0.5 * z_oil + 0.3 * z_dist + 0.2 * z_gold

    return (
        df.reset_index()
        .sort_values("Peace Dividend Score", ascending=False)
        .reset_index(drop=True)
    )
