from src.data_fetcher import fetch_combined_news, fetch_stock_data
from src.sentiment import analyze_news_sentiment


def run_fused_pipeline(ticker_symbol: str):
  print(f"\n==========================================")
  print(f"   RUNNING DATA PIPELINE FOR: {ticker_symbol}")
  print(f"==========================================\n")

  # 1. Fetch Price Data & Company Info
  stock_info = fetch_stock_data(ticker_symbol)
  print(f"[+] Market Ingestion Complete:")
  print(f"    - Company Name: {stock_info['company_name']}")
  print(f"    - Latest Close Price: ${stock_info['latest_price']}")
  print(f"    - Today's Return: {stock_info['daily_return_pct']:+.2f}%")
  print(f"    - Daily Volatility Spread: ${stock_info['volatility_spread']:.2f}\n")

  # 2. Ingest Multi-Source Headlines (RSS + yfinance)
  print(f"[+] Scraping headlines from multi-source feed...")
  raw_news = fetch_combined_news(ticker_symbol, stock_info["company_name"])
  print(f"[+] Total raw headlines ingested: {len(raw_news)}\n")

  # 3. Process FinBERT Sentiment, Source Tiering, & Time Decay
  print(f"[+] Running AI Engine & Calculating Weighted Index...")
  sentiment_results = analyze_news_sentiment(raw_news)

  S = sentiment_results["weighted_net_sentiment"]
  R = stock_info["daily_return_pct"]
  total_analyzed = sentiment_results["total_headlines_analyzed"]

  print(f"\n==========================================")
  print(f"   FINAL FUSED PIPELINE INSIGHTS")
  print(f"==========================================")
  print(f"Deduplicated Headlines Processed: {total_analyzed}")
  print(
      f"Weighted Net Sentiment Index (S): {S:+.4f} (-1.0 Bearish, +1.0 Bullish)"
  )
  print(f"Actual Daily Price Return (R):  {R:+.2f}%\n")

  # 4. Divergence Logic Check
  if S > 0.10 and R < -0.50:
    print(
        "⚠️ BULLISH DIVERGENCE DETECTED: Media sentiment is positive, but"
        " market price is DOWN."
    )
    print(
        "   -> Implied Bias: Profit-taking or broader market sell-off"
        " overriding headline tone."
    )
  elif S < -0.10 and R > 0.50:
    print(
        "⚠️ BEARISH DIVERGENCE DETECTED: Media sentiment is negative, but"
        " market price is UP."
    )
    print(
        "   -> Implied Bias: Bad news was likely already priced in by"
        " investors."
    )
  elif S > 0.10 and R > 0.50:
    print(
        "✅ ALIGNED BULLISH: Positive news flow is actively driving price gains."
    )
  elif S < -0.10 and R < -0.50:
    print(
        "✅ ALIGNED BEARISH: Negative news consensus is actively driving price"
        " sell-off."
    )
  else:
    print(
        "ℹ️ NEUTRAL: Market price move or headline sentiment is currently"
        " within neutral bounds."
    )


if __name__ == "__main__":
  run_fused_pipeline("TSLA")