import math
import re
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Tier 1: Top Institutional Financial Outlets (W_source = 1.0)
TIER_1_SOURCES = {
    "reuters",
    "bloomberg",
    "wall street journal",
    "wsj",
    "financial times",
    "cnbc",
    "barron's",
}

# Tier 2: Major Financial Media & Aggregators (W_source = 0.7)
TIER_2_SOURCES = {
    "yahoo finance",
    "marketwatch",
    "seeking alpha",
    "business insider",
    "motley fool",
    "benzinga",
    "investor's business daily",
    "fox business",
    "investopedia",
}

# Load FinBERT model globally
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def extract_domain_from_url(url: str) -> str:
  """Extracts domain name from URL if publisher string is missing or ambiguous."""
  if not url:
    return ""
  match = re.search(r"https?://(?:www\.)?([^/]+)", url)
  return match.group(1).lower() if match else ""


def get_source_weight(publisher_name: str, url: str = "") -> float:
  """Maps publisher name or domain to credibility weight W_source."""
  clean_publisher = (publisher_name or "").lower().strip()
  domain = extract_domain_from_url(url)

  if any(s in clean_publisher or s in domain for s in TIER_1_SOURCES):
    return 1.0
  elif any(s in clean_publisher or s in domain for s in TIER_2_SOURCES):
    return 0.7
  elif clean_publisher or domain:
    return 0.5
  else:
    return 0.3


def calculate_time_weight(pub_timestamp, half_life_hours: float = 6.0) -> float:
  """Calculates exponential time-decay factor W_time = e^(-lambda * delta_t)."""
  from datetime import datetime, timezone

  now = datetime.now(timezone.utc)

  if pub_timestamp.tzinfo is None:
    pub_timestamp = pub_timestamp.replace(tzinfo=timezone.utc)

  elapsed_seconds = (now - pub_timestamp).total_seconds()
  elapsed_hours = max(0.0, elapsed_seconds / 3600.0)

  decay_lambda = math.log(2) / half_life_hours
  w_time = math.exp(-decay_lambda * elapsed_hours)

  return round(w_time, 4)


def is_duplicate_headline(
    new_headline: str, existing_headlines: list[str], threshold: float = 0.70
) -> bool:
  """Checks Jaccard similarity between word sets to strip duplicate news."""
  clean = lambda txt: set(re.sub(r"[^\w\s]", "", txt.lower()).split())
  new_words = clean(new_headline)

  if not new_words:
    return False

  for existing in existing_headlines:
    existing_words = clean(existing)
    if not existing_words:
      continue

    intersection = new_words.intersection(existing_words)
    union = new_words.union(existing_words)

    if len(intersection) / len(union) >= threshold:
      return True

  return False


def analyze_news_sentiment(news_items: list[dict]) -> dict:
  """Executes FinBERT scoring, applies source/time weights, and calculates

  Non-Diluted Directional Weighted Sentiment Index (S_adj).
  """
  processed_records = []
  unique_headlines = []

  # Setup pipeline for probability distribution extraction
  multi_label_pipeline = pipeline(
      "sentiment-analysis",
      model=model,
      tokenizer=tokenizer,
      top_k=None,
  )

  for item in news_items:
    headline = item.get("title", "")

    if not headline or is_duplicate_headline(headline, unique_headlines):
      continue
    unique_headlines.append(headline)

    # Step A: Run FinBERT over headline
    results = multi_label_pipeline(headline[:512])[0]
    scores_dict = {res["label"].lower(): res["score"] for res in results}

    pos_score = scores_dict.get("positive", 0.0)
    neg_score = scores_dict.get("negative", 0.0)

    raw_delta = pos_score - neg_score

    top_label = max(scores_dict, key=scores_dict.get).upper()
    top_confidence = scores_dict[top_label.lower()]

    # Step B: Calculate weights
    w_source = get_source_weight(
        item.get("publisher", ""), item.get("link", "")
    )
    w_time = calculate_time_weight(item["pub_timestamp"])
    combined_weight = w_source * w_time

    processed_records.append({
        "Timestamp (UTC)": item["pub_timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        "Publisher": item.get("publisher", "Unknown"),
        "Headline": headline,
        "FinBERT Classification": top_label,
        "Confidence": f"{round(top_confidence * 100, 1)}%",
        "Raw Delta": round(raw_delta, 3),
        "Source Weight": w_source,
        "Time Decay Weight": w_time,
        "Combined Weight": round(combined_weight, 3),
    })

  df = pd.DataFrame(processed_records)

  if df.empty:
    return {
        "weighted_net_sentiment": 0.0,
        "total_headlines_analyzed": 0,
        "directional_headlines_analyzed": 0,
        "records_df": df,
    }

  # Filter directional headlines (POSITIVE and NEGATIVE)
  directional_df = df[df["FinBERT Classification"] != "NEUTRAL"]

  total_count = len(df)
  directional_count = len(directional_df)

  if directional_count > 0:
    dir_weighted_sum = (
        directional_df["Raw Delta"] * directional_df["Combined Weight"]
    ).sum()
    dir_weight_denom = directional_df["Combined Weight"].sum()

    s_dir = (
        dir_weighted_sum / dir_weight_denom if dir_weight_denom > 0 else 0.0
    )

    directional_ratio = directional_count / total_count
    s_adj = s_dir * math.sqrt(directional_ratio)
  else:
    s_adj = 0.0

  return {
      "weighted_net_sentiment": round(s_adj, 4),
      "total_headlines_analyzed": total_count,
      "directional_headlines_analyzed": directional_count,
      "records_df": df,
  }


def evaluate_quant_divergence(
    s_index: float,
    daily_return_pct: float,
    volatility_spread: float,
    latest_price: float,
) -> dict:
  """Universal divergence evaluator using Volatility-Normalized Return (R_norm)."""
  # 1. Compute Intraday Volatility % Spread (Daily Range relative to Price)
  volatility_pct = (
      (volatility_spread / latest_price) * 100 if latest_price > 0 else 1.0
  )

  # 2. Compute Volatility-Normalized Return (R_norm)
  r_norm = daily_return_pct / volatility_pct if volatility_pct > 0 else 0.0

  # 3. Universal Directional Thresholds
  if s_index > 0.08 and r_norm < -0.15:
    signal = "⚠️ BULLISH DIVERGENCE"
    desc = (
        "Good news is coming out, but the stock price hasn't moved up yet."
        " Watch for potential buying opportunities."
    )
    status = "warning"
  elif s_index < -0.08 and r_norm > 0.15:
    signal = "⚠️ BEARISH DIVERGENCE"
    desc = (
        "The stock price is climbing despite negative news flow. Watch out for"
        " a potential pullback."
    )
    status = "info"
  elif s_index > 0.08 and r_norm > 0.15:
    signal = "✅ ALIGNED BULLISH"
    desc = (
        "Positive news coverage is supported by a strong, steady upward price"
        " push."
    )
    status = "success"
  elif s_index < -0.08 and r_norm < -0.15:
    signal = "🚨 ALIGNED BEARISH"
    desc = (
        "Negative news coverage is confirmed by steady downward selling"
        " pressure."
    )
    status = "error"
  else:
    signal = "ℹ️ NEUTRAL / NO CLEAR SIGNAL"
    desc = (
        "Sentiment or price movement is too small to show a clear direction."
    )
    status = "neutral"

  return {
      "signal": signal,
      "description": desc,
      "status": status,
      "volatility_pct": round(volatility_pct, 2),
      "r_norm": round(r_norm, 3),
  }