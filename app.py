from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.data_fetcher import fetch_combined_news, fetch_stock_data
from src.sentiment import analyze_news_sentiment, evaluate_quant_divergence

# Load logo as PIL Image for browser tab favicon
favicon = Image.open("assets/logo.png")

# Page Configuration (Sets custom logo on browser tab)
st.set_page_config(
    page_title="Market Sentiment & Divergence Engine",
    page_icon=favicon,
    layout="wide",
)

st.title("📈 Quantitative Market Sentiment & Divergence Engine")
st.markdown(
    "Real-time corporate news sentiment paired with price returns to detect"
    " market divergences."
)

# Initialize Session State
if "analysis_data" not in st.session_state:
  st.session_state["analysis_data"] = None

if "ticker_history" not in st.session_state:
  st.session_state["ticker_history"] = []

# Sidebar Configuration
st.sidebar.header("Configuration")
ticker_input = (
    st.sidebar.text_input("Enter Stock Ticker:", "NVDA").upper().strip()
)
run_analysis = st.sidebar.button(
    "Analyze Stock Sentiment & Divergence", type="primary"
)


# Function to Update Search History (Max 3, Deduplicated, FIFO)
def update_ticker_history(ticker: str):
  history = st.session_state["ticker_history"]
  if ticker in history:
    history.remove(ticker)  # Remove existing to bring to top
  history.insert(0, ticker)  # Insert at front
  st.session_state["ticker_history"] = history[:3]  # Keep only recent 3


# Render Dynamic Recent Searches in Sidebar
clicked_recent_ticker = None
if st.session_state["ticker_history"]:
  st.sidebar.markdown("---")
  st.sidebar.subheader("Recent Searches")

  # Tightly aligned container columns that scale smoothly across 1, 2, or 3 buttons
  hist_cols = st.sidebar.columns(len(st.session_state["ticker_history"]))
  for idx, hist_ticker in enumerate(st.session_state["ticker_history"]):
    if hist_cols[idx].button(
        hist_ticker, key=f"hist_btn_{hist_ticker}", use_container_width=True
    ):
      clicked_recent_ticker = hist_ticker

# Determine Active Ticker
active_ticker = clicked_recent_ticker if clicked_recent_ticker else ticker_input
should_run = run_analysis or (clicked_recent_ticker is not None)

if should_run:
  if not active_ticker:
    st.warning("⚠️ Please enter a valid stock ticker symbol.")
  else:
    error_message = None
    stock_info = None
    sentiment_results = None

    # Ingestion & model inference inside spinner
    with st.spinner(
        f"Ingesting market data & scraping headlines for {active_ticker}..."
    ):
      try:
        stock_info = fetch_stock_data(active_ticker)
        raw_news = fetch_combined_news(
            active_ticker, stock_info["company_name"]
        )
        sentiment_results = analyze_news_sentiment(raw_news)
      except Exception as e:
        error_message = str(e)

    # UI updates and exception handling outside spinner block to ensure loading state unmounts
    if error_message:
      st.error(
          f"❌ **Invalid Ticker or Data Error:** '{active_ticker}' could not be"
          " fetched. Please verify the symbol and try again."
      )
      st.caption(f"Details: {error_message}")
      st.session_state["analysis_data"] = None
    else:
      st.session_state["analysis_data"] = {
          "stock_info": stock_info,
          "sentiment_results": sentiment_results,
      }
      update_ticker_history(active_ticker)
      st.rerun()

# Render Results Dashboard
if st.session_state["analysis_data"] is not None:
  data = st.session_state["analysis_data"]
  stock_info = data["stock_info"]
  sentiment_results = data["sentiment_results"]

  # Core Quant Divergence Calculation
  S = sentiment_results["weighted_net_sentiment"]
  R = stock_info["daily_return_pct"]
  vol_spread = stock_info["volatility_spread"]
  price = stock_info["latest_price"]

  divergence_result = evaluate_quant_divergence(
      s_index=S,
      daily_return_pct=R,
      volatility_spread=vol_spread,
      latest_price=price,
  )

  # Top Metric Banner
  col1, col2, col3, col4 = st.columns(4)
  col1.metric(
      "Ticker / Company", stock_info["ticker"], stock_info["company_name"]
  )
  col2.metric("Latest Price", f"${stock_info['latest_price']}")
  col3.metric(
      "Daily Return",
      f"{stock_info['daily_return_pct']}%",
      delta=f"{stock_info['daily_return_pct']}%",
  )
  col4.metric(
      "Weighted Sentiment Index (S_adj)",
      f"{S:.4f}",
      help=(
          "Directional non-diluted sentiment score taking into account"
          " source/time weights."
      ),
  )

  st.divider()

  # Core Signals
  st.subheader("Market Alignment & Divergence Analysis")

  left_col, right_col = st.columns([1.3, 1])

  with left_col:
    # Signal Display Box
    status = divergence_result["status"]
    msg = (
        f"**{divergence_result['signal']}**\n\n{divergence_result['description']}"
    )

    if status == "success":
      st.success(msg)
    elif status == "warning":
      st.warning(msg)
    elif status == "info":
      st.info(msg)
    elif status == "error":
      st.error(msg)
    else:
      st.info(f"ℹ️ **NEUTRAL / NO CLEAR DIVERGENCE:**\n\n{msg}")

    # Quant Stats Summary
    st.markdown(
        f"**Total / Directional Headlines:**"
        f" {sentiment_results['total_headlines_analyzed']} /"
        f" {sentiment_results['directional_headlines_analyzed']}"
    )
    st.markdown(
        f"**Daily Volatility Spread:** ${stock_info['volatility_spread']} ("
        f" **{divergence_result['volatility_pct']}%** of share price)"
    )
    st.markdown(
        f"**Volatility-Normalized Return ($R_{{norm}}$):**"
        f" `{divergence_result['r_norm']}`"
    )

  with right_col:
    # Sentiment Gauge Meter (Top margin expanded to t=70)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=S,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": "Non-Diluted Sentiment Index (S)",
                "font": {"size": 14},
            },
            gauge={
                "axis": {"range": [-1.0, 1.0], "tickwidth": 1},
                "bar": {
                    "color": (
                        "#00d46a"
                        if S > 0.08
                        else ("#ff4b4b" if S < -0.08 else "#1f77b4")
                    )
                },
                "steps": [
                    {"range": [-1.0, -0.08], "color": "rgba(255, 75, 75, 0.3)"},
                    {
                        "range": [-0.08, 0.08],
                        "color": "rgba(200, 200, 200, 0.3)",
                    },
                    {"range": [0.08, 1.0], "color": "rgba(0, 212, 106, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": S,
                },
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=30, r=30, t=70, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # User Guide Expander
    with st.expander("📖 How to read these metrics"):
      st.markdown("""
            * **Sentiment Index ($S_{\text{adj}}$):** Non-diluted FinBERT sentiment score ranging from **-1.0** to **+1.0**.
            * **Volatility-Normalized Return ($R_{\text{norm}}$):** Daily percentage return divided by intraday volatility spread. Indicates what % of total daily price movement was directional.
            * 🟢 **Green (> 0.08):** Dominant bullish news flow.
            * ⚪ **Gray (-0.08 to 0.08):** Neutral / balanced news flow.
            * 🔴 **Red (< -0.08):** Dominant bearish news flow.
            """)

  st.divider()

  # Ingested Headline Explorer Table
  st.subheader("Ingested Headline Explorer")
  df_display = sentiment_results["records_df"]

  if not df_display.empty:
    st.dataframe(
        df_display[[
            "Timestamp (UTC)",
            "Publisher",
            "Headline",
            "FinBERT Classification",
            "Confidence",
            "Source Weight",
            "Time Decay Weight",
            "Combined Weight",
        ]],
        use_container_width=True,
    )
  else:
    st.warning("No valid news items were ingested.")