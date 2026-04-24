import numpy as np
import pandas as pd


def _apply_weighting(method, chosen, signals, vols, softmax_alpha=4.0):
    """Return a normalised weight Series for the chosen tickers."""
    s = signals.loc[chosen]
    v = vols.loc[chosen]

    if method == "Equal Weight":
        w = pd.Series(1.0, index=chosen)

    elif method == "Rank Weight":
        ranks = pd.Series(range(1, len(chosen) + 1), index=chosen[::-1])  # best signal = highest rank
        w = ranks.astype(float)

    elif method == "Softmax":
        exp_s = np.exp(softmax_alpha * s)
        w = exp_s

    else:  # Inverse Volatility (default)
        inv_vol = 1.0 / v.replace(0, np.nan)
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
        w = inv_vol

    w = w.replace([np.inf, -np.inf], np.nan).dropna()
    total = w.sum()
    if total == 0 or len(w) == 0:
        return None
    return w / total


def compute_target_weights(
    prices,
    tickers,
    momentum_lookback_days=126,
    vol_lookback_days=20,
    top_n=4,
    rebalance_freq="ME",
    use_spy_filter=True,
    regime_filter_ticker="SPY",
    regime_filter_ma_days=200,
    weight_change_threshold=0.02,
    weighting_method="Inverse Volatility",
    softmax_alpha=4.0,
):
    px = prices[tickers].copy()
    rets = px.pct_change()

    momentum = px / px.shift(momentum_lookback_days) - 1.0
    vol = rets.rolling(vol_lookback_days).std()
    signal = momentum / vol.replace(0, np.nan)

    signal_m = signal.resample(rebalance_freq).last()
    vol_m = vol.resample(rebalance_freq).last()

    weights = pd.DataFrame(0.0, index=signal_m.index, columns=signal_m.columns)

    for dt in signal_m.index:
        s = signal_m.loc[dt].dropna()
        v = vol_m.loc[dt].dropna()

        s = s[s > 0]
        if len(s) == 0:
            continue

        chosen = s.sort_values(ascending=False).head(top_n).index.tolist()
        w = _apply_weighting(weighting_method, chosen, s, v, softmax_alpha)
        if w is None:
            continue
        weights.loc[dt, w.index] = w.values

    if use_spy_filter and regime_filter_ticker in prices.columns:
        regime = prices[regime_filter_ticker]
        regime_ma = regime.rolling(regime_filter_ma_days).mean()
        risk_on = (regime > regime_ma).resample(rebalance_freq).last()
        risk_on = risk_on.reindex(weights.index).fillna(False)
        weights = weights.mul(risk_on.astype(float), axis=0)

    out = weights.copy()
    prev = pd.Series(0.0, index=weights.columns)

    for dt in weights.index:
        target = weights.loc[dt].copy()
        delta = (target - prev).abs()
        keep_prev = delta < weight_change_threshold
        target.loc[keep_prev] = prev.loc[keep_prev]

        total = target.sum()
        if total > 0:
            target = target / total

        out.loc[dt] = target
        prev = target

    return out.fillna(0.0)
