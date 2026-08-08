# 📈 Quantitative Financial Sentiment & Market Divergence Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FinBERT-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An automated financial NLP pipeline that ingests multi-source market news, applies deep-learning sentiment classification via **FinBERT**, weights stories based on **source credibility** and **exponential time-decay**, and cross-references headline sentiment with volatility-normalized price returns to detect **market divergences**.

---

## 💡 Overview

Market price movements are frequently driven by corporate news flow. However, standard headline sentiment scores often fail because of neutral noise dilution and asset volatility bias.

This engine solves this problem by calculating a **Non-Diluted Weighted Sentiment Index ($S_{\text{adj}}$)** bounded between **$-1.0$ (Extreme Bearish)** and **$+1.0$ (Extreme Bullish)**. It then compares this index against the stock's **Volatility-Normalized Return ($R_{\text{norm}}$)** to flag underlying market biases:

- **Bullish Divergence ($\mathbf{S > 0.08, R_{\text{norm}} < -0.15}$):** Positive news flow is meeting temporary selling pressure, signaling potential institutional accumulation.
- **Bearish Divergence ($\mathbf{S < -0.08, R_{\text{norm}} > 0.15}$):** Price is advancing despite negative news flow, warning of a potential bull trap or profit-taking.
- **Aligned Bullish / Bearish ($\mathbf{|S| > 0.08, |R_{\text{norm}}| > 0.15}$):** News sentiment matches clean directional price momentum.

---

## ✨ Key Features

- **Multi-Source Scraping:** Combines `yfinance` ticker news with **Google News RSS feeds** using unthrottled parameter encoding to capture a dense headline sample without hit limits.
- **Jaccard Headline Deduplication:** Filters out redundant or syndicated stories across multiple outlets using a token similarity threshold ($>0.70$).
- **Neutral Headline Noise Solution:** Scales directional sentiment ($S_{\text{dir}}$) by the square root of directional density ($\sqrt{N_{\text{dir}} / N_{\text{total}}}$) so non-directional or factual headlines do not artificially dilute sentiment intensity.
- **Source Credibility Tiering:**
  - **Tier 1 ($W_{\text{source}} = 1.0$):** Major institutional media (*Reuters, Bloomberg, Wall Street Journal, Financial Times, CNBC, Barron's*).
  - **Tier 2 ($W_{\text{source}} = 0.7$):** Major aggregators (*Yahoo Finance, MarketWatch, Seeking Alpha, Business Insider, Benzinga*).
  - **Tier 3 / Neutral ($W_{\text{source}} = 0.5$ / $0.3$):** Regional news, retail blogs, or unmapped sources with domain fallback verification.
- **Exponential Recency Decay ($W_{\text{time}}$):** Applies a 6-hour half-life exponential decay factor ($W_{\text{time}} = e^{-\lambda \cdot \Delta t}$) so breaking news carries significantly more weight than legacy stories.
- **Asset Volatility Normalization ($R_{\text{norm}}$):** Divides close-to-close percentage return by the intraday volatility spread relative to share price, creating a universal baseline across high-beta equities, low-beta stocks, and index ETFs.
- **Interactive Streamlit Web Interface:** Live dashboard featuring a non-diluted sentiment gauge, real-time metrics, interactive recent search history, and an audited headline explorer.