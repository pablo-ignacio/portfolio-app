import numpy as np
import pandas as pd


def max_drawdown(cum_curve):
    running_max = cum_curve.cummax()
    drawdown = cum_curve / running_max - 1.0
    return float(drawdown.min())


def run_backtest(
    prices,
    target_weights,
    tickers,
    vol_lookback_days=20,
    annual_target_vol=0.12,
    max_leverage=1.5,
    transaction_cost_bps=5.0,
):
    returns = prices[tickers].pct_change().dropna(how="all")

    weights_daily = target_weights.reindex(returns.index).ffill().fillna(0.0)
    weights_daily = weights_daily.shift(1).fillna(0.0)

    portfolio_gross = (weights_daily * returns).sum(axis=1)

    if annual_target_vol is not None:
        realized_vol = portfolio_gross.rolling(vol_lookback_days).std() * np.sqrt(252)
        scale = annual_target_vol / realized_vol.replace(0, np.nan)
        scale = scale.clip(upper=max_leverage)
        scale = scale.shift(1).fillna(1.0)
        portfolio_gross = portfolio_gross * scale

    turnover = target_weights.diff().abs().sum(axis=1).fillna(target_weights.abs().sum(axis=1))
    tc_per_rebalance = turnover * (transaction_cost_bps / 10000.0)

    tc_daily = pd.Series(0.0, index=returns.index)
    for dt in target_weights.index:
        future_dates = tc_daily.index[tc_daily.index >= dt]
        if len(future_dates) > 0:
            tc_daily.loc[future_dates[0]] += tc_per_rebalance.loc[dt]

    portfolio_net = portfolio_gross - tc_daily

    r = portfolio_net.dropna()
    cum = (1.0 + r).cumprod()

    ann_return = (1.0 + r).prod() ** (252 / len(r)) - 1.0
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    mdd = max_drawdown(cum)

    downside = r[r < 0].std() * np.sqrt(252)
    sortino = ann_return / downside if downside > 0 else np.nan
    calmar = ann_return / abs(mdd) if mdd != 0 else np.nan

    avg_turnover = turnover.mean()
    avg_holdings = (target_weights > 0).sum(axis=1).mean()

    rolling_window = 126  # ~6 months
    rolling_sharpe = (
        r.rolling(rolling_window).mean() /
        r.rolling(rolling_window).std()
    ) * np.sqrt(252)

    performance = {
        "Annual Return": ann_return,
        "Annual Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": mdd,
        "Hit Rate": float((r > 0).mean()),
        "Sortino": sortino,
        "Calmar": calmar,
        "Avg Monthly Turnover": avg_turnover,
        "Avg Holdings": avg_holdings,
    }

    return {
        "returns": r,
        "cum_returns": cum,
        "weights_daily": weights_daily,
        "performance": performance,
        "rolling_sharpe": rolling_sharpe,
    }
