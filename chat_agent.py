# chat_agent.py — AI assistant backed by OpenAI Responses API
#
# Tools available to the model:
#   1. web_search_preview  — built-in OpenAI web search (news, real-world context)
#   2. compute_statistic   — executes Python on the live prices/returns DataFrames
#
# The agentic loop keeps running until the model produces a final text response
# with no further tool calls pending.

import io
import json
import builtins
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for server use
import matplotlib.pyplot as plt
import scipy.stats as stats
import yfinance as yf
from sklearn.linear_model import LinearRegression
from openai import OpenAI


def _fetch_series(ticker: str, start=None, end=None) -> pd.Series:
    """
    Fetch a single price series from yfinance and return it as a clean pd.Series.
    Handles both flat and MultiIndex column structures across yfinance versions.
    """
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    if isinstance(data.columns, pd.MultiIndex):
        series = data["Close"].iloc[:, 0]
    else:
        series = data["Close"]
    series.name = ticker
    return series


_SYSTEM = """\
You are a financial analysis assistant embedded in a portfolio analysis app.
You have access to historical ETF price data for the following tickers: {tickers}.
The data spans from {start} to {end}.

Two variables are always available for computation:
  - prices   : pd.DataFrame of adjusted close prices (dates × tickers)
  - returns  : pd.DataFrame of daily percentage returns (same shape)

Pre-imported (no import needed):
  - np       : numpy
  - pd       : pandas
  - plt      : matplotlib.pyplot
  - stats    : scipy.stats  (OLS, t-tests, normality, etc.)
  - LinearRegression : sklearn.linear_model.LinearRegression
  - fetch_series(ticker, start, end) → pd.Series  (price/rate from yfinance)

You may also `import` any installed library (e.g. statsmodels).

FETCHING EXTRA DATA
If the user asks about a series not in `prices`, use fetch_series — do NOT call
yf.download directly:

  rf = fetch_series("^IRX", start=prices.index[0], end=prices.index[-1])
  rf = rf.reindex(prices.index).ffill() / 100 / 252   # annualise → daily

Always .reindex(prices.index).ffill() to align dates before computing anything.

IMPORTANT: before accessing any ticker in `prices` or `returns`, check whether it is
present: `if 'SLX' not in prices.columns`. If it is missing, fetch it with
fetch_series and compute its returns manually: `r = s.pct_change().dropna()`.
Never assume a ticker is in `prices` just because it was mentioned or plotted earlier.

Common tickers:
  Rates/macro : ^IRX (3M T-bill / risk-free rate), ^TNX (10Y yield), ^TYX (30Y yield),
                DX-Y.NYB (USD index), EURUSD=X, JPYUSD=X
  Commodities : HG=F (copper), CL=F (WTI oil), BZ=F (Brent), GC=F (gold futures),
                SI=F (silver), NG=F (natural gas), ZC=F (corn), ZW=F (wheat)
  Crypto      : BTC-USD, ETH-USD
  Volatility  : ^VIX
  Note        : GLD and ^VIX are already in `prices` — no fetch needed for either.

STATISTICAL ANALYSES
Always use compute_statistic to run code — never just show it without executing.

  OLS with full output  : import statsmodels.api as sm
                          res = sm.OLS(y, sm.add_constant(X)).fit()
                          result = res.summary().as_text()

  Linear regression     : stats.linregress(x, y)  or  LinearRegression()

  Polynomial regression : np.polyfit(x, y, deg=2); np.polyval(coeffs, x)

  Rolling correlation   : s1.rolling(window).corr(s2)

  Lagged regression     : import statsmodels.api as sm
                          df = pd.DataFrame({{'y': prices['COPX'].pct_change(),
                                             'x': prices['^VIX'],
                                             'x_lag1': prices['^VIX'].shift(1)}}).dropna()
                          res = sm.OLS(df['y'], sm.add_constant(df[['x','x_lag1']])).fit()
                          result = res.summary().as_text()

  Granger causality     : from statsmodels.tsa.stattools import grangercausalitytests

  Cointegration         : from statsmodels.tsa.stattools import coint

RULES
- For ANY quantitative task use compute_statistic. Always run code, never just show it.
- For charts: build with plt, set result = plt.gcf().
- For news / macro context: use web_search.
- Format numbers clearly: "beta = 0.82", "R² = 0.67", "p = 0.003".
- Be concise but complete.
"""

_TOOLS = [
    {"type": "web_search_preview"},
    {
        "type": "function",
        "name": "compute_statistic",
        "description": (
            "Execute Python code against the live ETF dataset and return the result. "
            "Pre-available without import: prices, returns, np, pd, plt, stats, "
            "LinearRegression, fetch_series. "
            "Standard imports also work (e.g. import statsmodels.api as sm). "
            "MUST assign the final answer to a variable named 'result'. "
            "Always run code — never just show it without executing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Valid Python that assigns to 'result'.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

# Globals exposed to model-generated code
_EXEC_GLOBALS = {
    "__builtins__": builtins,
    "np":               np,
    "pd":               pd,
    "plt":              plt,
    "stats":            stats,
    "LinearRegression": LinearRegression,
    "fetch_series":     _fetch_series,
    "yf":               yf,
}


def _exec_code(
    code: str, prices: pd.DataFrame, returns: pd.DataFrame
) -> str | bytes:
    """
    Execute model-generated code with full builtins and statistical libraries.

    Returns
    -------
    str   — text result (or error message)
    bytes — PNG image bytes if the code produced a matplotlib figure
    """
    plt.close("all")
    local_ns = {
        "prices":  prices,
        "returns": returns,
        "result":  None,
    }
    try:
        exec(code, _EXEC_GLOBALS, local_ns)
        result = local_ns.get("result")

        fig = None
        if isinstance(result, plt.Figure):
            fig = result
        elif plt.get_fignums():
            fig = plt.gcf()

        if fig is not None:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)
            plt.close("all")
            return buf.getvalue()

        if result is None:
            return "Code ran but 'result' was never assigned."
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return result.to_string()
        return str(result)
    except Exception as exc:
        plt.close("all")
        return f"Execution error: {exc}"


def run_agent(
    question: str,
    prices: pd.DataFrame,
    history: list[tuple[str, str]],
    api_key: str,
) -> tuple[str, list[bytes]]:
    """
    Run one question through the agent.

    Returns
    -------
    (answer_text, images)
      answer_text : str         — the model's final plain-English response
      images      : list[bytes] — PNG bytes for any plots produced (may be empty)
    """
    client  = OpenAI(api_key=api_key)
    returns = prices.pct_change().dropna(how="all")

    system = _SYSTEM.format(
        tickers=", ".join(prices.columns.tolist()),
        start=str(prices.index[0].date()),
        end=str(prices.index[-1].date()),
    )

    input_items: list = []
    for user_msg, assistant_msg in history[-6:]:
        input_items.append({"role": "user",      "content": user_msg})
        input_items.append({"role": "assistant", "content": assistant_msg})
    input_items.append({"role": "user", "content": question})

    images: list[bytes] = []

    while True:
        response = client.responses.create(
            model="gpt-4o",
            instructions=system,
            tools=_TOOLS,
            input=input_items,
        )

        for item in response.output:
            input_items.append(item.model_dump())

        pending = [item for item in response.output if item.type == "function_call"]

        if not pending:
            break

        for fc in pending:
            if fc.name == "compute_statistic":
                code   = json.loads(fc.arguments).get("code", "")
                output = _exec_code(code, prices, returns)
            else:
                output = "Unknown tool."

            if isinstance(output, bytes):
                images.append(output)
                tool_text = "Plot generated successfully — it will be shown to the user."
            else:
                tool_text = output

            input_items.append({
                "type":    "function_call_output",
                "call_id": fc.call_id,
                "output":  tool_text,
            })

    return response.output_text, images
