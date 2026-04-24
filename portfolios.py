# portfolios.py — portfolio construction methods for comparison dashboard

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Ledoit-Wolf robust covariance estimator
try:
    from sklearn.covariance import LedoitWolf
    _HAS_LEDOIT = True
except ImportError:
    _HAS_LEDOIT = False


# ── Covariance helper ─────────────────────────────────────────────────────────

def _estimate_cov(returns_window: pd.DataFrame) -> np.ndarray:
    """
    Estimate covariance matrix.
    Uses Ledoit-Wolf shrinkage if sklearn is available, else sample covariance.
    Falls back to sample covariance if Ledoit-Wolf fails.
    """
    X = returns_window.dropna().values
    if len(X) < returns_window.shape[1] + 2:
        # Not enough observations — return sample cov
        return np.cov(X.T) if len(X) > 1 else np.eye(returns_window.shape[1])
    if _HAS_LEDOIT:
        try:
            lw = LedoitWolf().fit(X)
            return lw.covariance_
        except Exception:
            pass  # fall back to sample cov
    return np.cov(X.T)


# ── Optimisation helpers ──────────────────────────────────────────────────────

def _min_variance_weights(cov: np.ndarray, n: int) -> np.ndarray | None:
    """Long-only minimum variance via scipy. Returns None on failure."""
    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n
    try:
        res = minimize(
            lambda w: w @ cov @ w,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if res.success:
            w = np.clip(res.x, 0, None)
            total = w.sum()
            return w / total if total > 0 else None
    except Exception:
        pass
    return None


def _mean_variance_weights(
    expected_returns: np.ndarray, cov: np.ndarray, n: int, risk_aversion: float = 3.0
) -> np.ndarray | None:
    """
    Long-only mean-variance (maximise expected_return - 0.5*risk_aversion*variance).
    Falls back to equal weight if optimisation fails.
    """
    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n

    def neg_utility(w):
        return -(w @ expected_returns - 0.5 * risk_aversion * (w @ cov @ w))

    try:
        res = minimize(
            neg_utility,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if res.success:
            w = np.clip(res.x, 0, None)
            total = w.sum()
            return w / total if total > 0 else None
    except Exception:
        pass
    return None


# ── SPY regime filter ─────────────────────────────────────────────────────────

def _apply_spy_filter(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    regime_ticker: str,
    ma_days: int,
    rebalance_freq: str,
) -> pd.DataFrame:
    if regime_ticker not in prices.columns:
        return weights
    regime = prices[regime_ticker]
    regime_ma = regime.rolling(ma_days).mean()
    risk_on = (regime > regime_ma).resample(rebalance_freq).last()
    risk_on = risk_on.reindex(weights.index).fillna(False)
    return weights.mul(risk_on.astype(float), axis=0)


def _smooth_weights(weights: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Suppress small weight changes to reduce turnover."""
    out = weights.copy()
    prev = pd.Series(0.0, index=weights.columns)
    for dt in weights.index:
        target = weights.loc[dt].copy()
        delta = (target - prev).abs()
        target[delta < threshold] = prev[delta < threshold]
        total = target.sum()
        if total > 0:
            target = target / total
        out.loc[dt] = target
        prev = target
    return out.fillna(0.0)


# ── Main entry point ──────────────────────────────────────────────────────────

def build_portfolio(
    method: str,
    prices: pd.DataFrame,
    tickers: list[str],
    momentum_lookback: int = 126,
    vol_lookback: int = 20,
    cov_lookback: int = 60,
    top_n: int = 10,
    rebalance_freq: str = "ME",
    use_spy_filter: bool = True,
    regime_ticker: str = "SPY",
    regime_ma_days: int = 200,
    weight_change_threshold: float = 0.02,
    softmax_alpha: float = 4.0,
    risk_aversion: float = 3.0,
) -> dict:
    """
    Build a portfolio for the given method.

    Returns
    -------
    dict with:
      "weights"       : pd.DataFrame (rebalance dates × tickers)
      "latest_weights": pd.Series    (latest rebalance)
      "returns"       : pd.Series    (daily portfolio returns, gross of vol scaling)
      "cum_returns"   : pd.Series
    """
    px = prices[tickers].copy()
    rets = px.pct_change()

    # Signals used by momentum-based methods
    momentum = px / px.shift(momentum_lookback) - 1.0
    vol = rets.rolling(vol_lookback).std()
    signal = momentum / vol.replace(0, np.nan)

    signal_m = signal.resample(rebalance_freq).last()
    vol_m = vol.resample(rebalance_freq).last()

    weights = pd.DataFrame(0.0, index=signal_m.index, columns=signal_m.columns)

    for dt in signal_m.index:
        s = signal_m.loc[dt].dropna()
        v = vol_m.loc[dt].dropna()
        common = s.index.intersection(v.index)
        s, v = s[common], v[common]

        if method in ("Inverse Volatility", "Equal Weight", "Rank Weight", "Softmax"):
            s_pos = s[s > 0]
            if len(s_pos) == 0:
                continue
            chosen = s_pos.sort_values(ascending=False).head(top_n).index.tolist()

            if method == "Equal Weight":
                w = pd.Series(1.0 / len(chosen), index=chosen)

            elif method == "Rank Weight":
                n_chosen = len(chosen)
                rank_vals = list(range(n_chosen, 0, -1))  # best = n, worst = 1
                w = pd.Series(rank_vals, index=chosen, dtype=float)
                w = w / w.sum()

            elif method == "Softmax":
                exp_s = np.exp(softmax_alpha * s[chosen])
                exp_s = exp_s.replace([np.inf, -np.inf], np.nan).dropna()
                if exp_s.sum() == 0:
                    continue
                w = exp_s / exp_s.sum()

            else:  # Inverse Volatility
                iv = 1.0 / v.loc[chosen].replace(0, np.nan)
                iv = iv.replace([np.inf, -np.inf], np.nan).dropna()
                if iv.sum() == 0:
                    continue
                w = iv / iv.sum()

        elif method in ("Min Variance", "Mean-Variance"):
            # Use top_n by signal as candidate universe (avoid huge matrices)
            s_pos = s[s > 0]
            candidates = (
                s_pos.sort_values(ascending=False).head(top_n).index.tolist()
                if len(s_pos) > 0
                else s.sort_values(ascending=False).head(top_n).index.tolist()
            )
            if len(candidates) < 2:
                continue

            # Rolling covariance window ending at dt
            hist = rets.loc[:dt, candidates].tail(cov_lookback)
            if len(hist) < len(candidates) + 2:
                # Fall back to equal weight — not enough history
                w = pd.Series(1.0 / len(candidates), index=candidates)
            else:
                cov_mat = _estimate_cov(hist)
                n_c = len(candidates)

                if method == "Min Variance":
                    w_arr = _min_variance_weights(cov_mat, n_c)
                else:  # Mean-Variance
                    mu = s[candidates].values  # use risk-adj signal as expected return proxy
                    w_arr = _mean_variance_weights(mu, cov_mat, n_c, risk_aversion)

                if w_arr is None:
                    # Fallback: equal weight
                    w_arr = np.ones(n_c) / n_c

                w = pd.Series(w_arr, index=candidates)

        else:
            continue

        w = w.clip(lower=0)
        total = w.sum()
        if total > 0:
            w = w / total
        weights.loc[dt, w.index] = w.values

    if use_spy_filter:
        weights = _apply_spy_filter(
            weights, prices, regime_ticker, regime_ma_days, rebalance_freq
        )

    weights = _smooth_weights(weights, weight_change_threshold)

    # Daily returns (gross, no vol scaling — kept simple for comparison)
    weights_daily = weights.reindex(rets.index).ffill().fillna(0.0)
    weights_daily = weights_daily.shift(1).fillna(0.0)
    port_returns = (weights_daily * rets).sum(axis=1).dropna()
    cum_returns = (1.0 + port_returns).cumprod()

    latest_rb = weights.index[weights.index <= rets.index[-1]][-1]
    latest_w = weights.loc[latest_rb]

    return {
        "weights": weights,
        "latest_weights": latest_w,
        "returns": port_returns,
        "cum_returns": cum_returns,
    }
