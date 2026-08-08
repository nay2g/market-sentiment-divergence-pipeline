import certifi
from datetime import datetime, timezone
import re
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import yfinance as yf


def fetch_stock_data(ticker_symbol: str) -> dict:
  """Fetches market price history, daily return %, and high/low volatility spread."""
  ticker = yf.Ticker(ticker_symbol)
  hist = ticker.history(period="5d", interval="1d")

  if hist.empty or len(hist) < 2:
    raise ValueError(
        f"Insufficient price history found for ticker '{ticker_symbol}'."
    )

  latest_close = hist["Close"].iloc[-1]
  prev_close = hist["Close"].iloc[-2]
  daily_return_pct = ((latest_close - prev_close) / prev_close) * 100.0

  day_high = hist["High"].iloc[-1]
  day_low = hist["Low"].iloc[-1]
  volatility_spread = day_high - day_low

  company_name = ticker.info.get("shortName", ticker_symbol)

  return {
      "ticker": ticker_symbol.upper(),
      "company_name": company_name,
      "latest_price": round(latest_close, 2),
      "daily_return_pct": round(daily_return_pct, 2),
      "volatility_spread": round(volatility_spread, 2),
  }


def parse_rss_pubdate(pub_date_str: str) -> datetime:
  """Parses standard RSS pubDate string into UTC datetime object."""
  try:
    from email.utils import parsedate_to_datetime

    dt = parsedate_to_datetime(pub_date_str)
    return dt.astimezone(timezone.utc)
  except Exception:
    return datetime.now(timezone.utc)


def fetch_google_news_rss(query: str, max_results: int = 40) -> list[dict]:
  """Fetches news headlines from Google News RSS feed securely using certifi."""
  encoded_query = urllib.parse.quote(query)
  rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      )
  }

  req = urllib.request.Request(rss_url, headers=headers)
  news_items = []

  # Create a secure default SSL context using certifi's CA bundle
  ssl_context = ssl.create_default_context(cafile=certifi.where())

  try:
    with urllib.request.urlopen(
        req, context=ssl_context, timeout=10
    ) as response:
      xml_data = response.read()

    root = ET.fromstring(xml_data)

    for item in root.findall(".//item")[:max_results]:
      title = item.find("title").text if item.find("title") is not None else ""
      link = item.find("link").text if item.find("link") is not None else ""
      pub_date = (
          item.find("pubDate").text
          if item.find("pubDate") is not None
          else None
      )

      publisher = "Google News"
      clean_title = title
      if " - " in title:
        parts = title.rsplit(" - ", 1)
        clean_title = parts[0].strip()
        publisher = parts[1].strip()

      pub_dt = (
          parse_rss_pubdate(pub_date)
          if pub_date
          else datetime.now(timezone.utc)
      )

      news_items.append({
          "title": clean_title,
          "publisher": publisher,
          "link": link,
          "pub_timestamp": pub_dt,
      })
  except Exception as e:
    print(f"Warning: Failed to fetch Google RSS news: {e}")

  return news_items


def fetch_combined_news(ticker_symbol: str, company_name: str) -> list[dict]:
  """Combines yfinance news and Google News RSS to reach 30-50 headlines."""
  combined = []

  # Source 1: Google News RSS
  search_query = f"{ticker_symbol} {company_name}"
  rss_items = fetch_google_news_rss(search_query, max_results=40)
  combined.extend(rss_items)

  # Source 2: yfinance news feed
  try:
    ticker = yf.Ticker(ticker_symbol)
    yf_news = ticker.news or []
    for item in yf_news:
      content = item.get("content", item)
      title = content.get("title", "")

      provider = content.get("provider", {})
      publisher = (
          provider.get("displayName")
          if isinstance(provider, dict)
          else item.get("publisher", "Yahoo Finance")
      )

      canonical = content.get("canonicalUrl", {})
      link = (
          canonical.get("url", "")
          if isinstance(canonical, dict)
          else item.get("link", "")
      )

      pub_date_str = content.get("pubDate")
      provider_time = item.get("providerPublishTime")

      if pub_date_str:
        try:
          from email.utils import parsedate_to_datetime

          pub_dt = parsedate_to_datetime(pub_date_str).astimezone(timezone.utc)
        except Exception:
          pub_dt = datetime.now(timezone.utc)
      elif provider_time:
        pub_dt = datetime.fromtimestamp(provider_time, tz=timezone.utc)
      else:
        pub_dt = datetime.now(timezone.utc)

      if title:
        combined.append({
            "title": title,
            "publisher": publisher or "Yahoo Finance",
            "link": link,
            "pub_timestamp": pub_dt,
        })
  except Exception as e:
    print(f"Warning: Failed to fetch yfinance news: {e}")

  return combined