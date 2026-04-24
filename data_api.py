import pandas as pd
import yfinance as yf
import streamlit as st


def get_prices(tickers, start_date="2015-01-01", end_date=None):
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers[:1]

    prices = prices.sort_index().ffill().dropna(how="all")

    # Retry any tickers that came back entirely empty (Yahoo Finance rate-limit drop).
    # Fetch them one at a time so a single bad ticker can't poison the whole batch.
    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    for t in missing:
        try:
            single = yf.download(
                tickers=t,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )
            if not single.empty:
                col = single["Close"] if "Close" in single.columns else single.iloc[:, 0]
                prices[t] = col.reindex(prices.index).ffill()
        except Exception:
            pass  # leave the ticker absent; downstream code handles NaN columns

    return prices


@st.cache_data(ttl=3600)
def get_etf_news_why(tickers: tuple) -> dict:
    """
    Fetch the most recent Yahoo Finance news headline + summary for each ticker.
    Returns a dict {ticker: "why string"}.
    Cached for 1 hour so it refreshes on page reload after the TTL expires,
    and always fetches fresh on the first load of a new session.
    """
    result = {}
    for ticker in tickers:
        try:
            news = yf.Ticker(ticker).news
            if not news:
                result[ticker] = "No recent news available."
                continue
            # Pick the most recent item with a usable summary
            for item in news[:5]:
                content = item.get("content", {})
                summary = (content.get("summary") or "").strip()
                title   = (content.get("title")   or "").strip()
                text = summary if len(summary) > 40 else title
                if text:
                    result[ticker] = text
                    break
            else:
                result[ticker] = "No recent news available."
        except Exception:
            result[ticker] = "Could not fetch news."
    return result
