"""
model_engine.py — FinBERT Sentiment Analysis Engine
=====================================================
Handles:
  1. FinBERT model loading with GPU auto-detection
  2. Batch sentiment inference (batch_size=32)
  3. Score aggregation at ticker, sector, and market breadth levels
"""

import logging
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from config import NIFTY50_TICKERS, TICKER_NAMES, MODEL_NAME, BATCH_SIZE

logger = logging.getLogger(__name__)

# =============================================================================
# 1. MODEL LOADING
# =============================================================================

def load_finbert_pipeline():
    """
    Initialize the FinBERT text-classification pipeline.
    - Auto-detects GPU (CUDA) and falls back to CPU.
    - Sets eval mode + disables gradients for faster inference.
    - Uses max_length=128 (headlines are short — no need for 512).
    - Returns the pipeline object for reuse.
    
    This function is designed to be wrapped with @st.cache_resource in app.py.
    """
    device = 0 if torch.cuda.is_available() else -1
    device_name = "CUDA GPU" if device == 0 else "CPU"
    logger.info(f"Loading FinBERT model '{MODEL_NAME}' on {device_name}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # Optimization: eval mode disables dropout, no_grad skips gradient tracking
    model.eval()
    torch.set_grad_enabled(False)

    sentiment_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,          # Return all 3 class probabilities
        device=device,
        batch_size=BATCH_SIZE,
        truncation=True,
        max_length=128,      # Headlines are short — 128 tokens is plenty
    )

    logger.info(f"FinBERT loaded successfully on {device_name}.")
    return sentiment_pipeline


# =============================================================================
# 2. BATCH SENTIMENT INFERENCE
# =============================================================================

def _parse_finbert_output(result: List[dict]) -> dict:
    """
    Parse the output of a single FinBERT prediction.
    FinBERT returns a list of dicts like:
      [{'label': 'positive', 'score': 0.85}, {'label': 'negative', 'score': 0.05}, ...]
    
    Returns dict with p_positive, p_negative, p_neutral, label, confidence, net_score.
    """
    probs = {item["label"]: item["score"] for item in result}

    p_positive = probs.get("positive", 0.0)
    p_negative = probs.get("negative", 0.0)
    p_neutral  = probs.get("neutral", 0.0)

    # Net score: P(positive) - P(negative)
    net_score = p_positive - p_negative

    # Dominant label
    label = max(probs, key=probs.get)
    confidence = probs[label]

    return {
        "p_positive": round(p_positive, 4),
        "p_negative": round(p_negative, 4),
        "p_neutral":  round(p_neutral, 4),
        "net_score":  round(net_score, 4),
        "label":      label,
        "confidence": round(confidence, 4),
    }


def analyze_sentiment(
    headlines: List[str], pipe
) -> pd.DataFrame:
    """
    Run batch FinBERT inference on a list of headlines.
    
    Returns DataFrame with columns:
        headline, label, confidence, p_positive, p_negative, p_neutral, net_score
    """
    if not headlines:
        return pd.DataFrame(columns=[
            "headline", "label", "confidence",
            "p_positive", "p_negative", "p_neutral", "net_score"
        ])

    # FinBERT batch inference
    raw_results = pipe(headlines)

    rows = []
    for headline, result in zip(headlines, raw_results):
        parsed = _parse_finbert_output(result)
        parsed["headline"] = headline
        rows.append(parsed)

    df = pd.DataFrame(rows)
    df = df[["headline", "label", "confidence",
             "p_positive", "p_negative", "p_neutral", "net_score"]]
    return df


# =============================================================================
# 3. TICKER-LEVEL AGGREGATION
# =============================================================================

def compute_ticker_sentiment(
    all_headlines: Dict[str, List[str]],
    pipe,
    tickers_sectors: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Process all headlines across every ticker via batch FinBERT inference.
    
    Args:
        all_headlines: {ticker: [headline, ...]}
        pipe: FinBERT pipeline
        tickers_sectors: {ticker: sector} mapping
    
    Returns:
        ticker_summary: DataFrame with one row per ticker
            Columns: ticker, company, sector, mean_net_score, pct_positive,
                     pct_negative, pct_neutral, article_count
        headline_details: {ticker: DataFrame of per-headline results}
    """
    if tickers_sectors is None:
        tickers_sectors = NIFTY50_TICKERS

    # Flatten all headlines for batch processing
    flat_headlines = []
    flat_tickers = []
    for ticker, headlines in all_headlines.items():
        for h in headlines:
            flat_headlines.append(h)
            flat_tickers.append(ticker)

    logger.info(f"Running FinBERT on {len(flat_headlines)} total headlines...")

    # Single batch inference call for maximum efficiency
    if flat_headlines:
        raw_results = pipe(flat_headlines)
    else:
        raw_results = []

    # Parse results and attach ticker info
    all_rows = []
    for ticker, headline, result in zip(flat_tickers, flat_headlines, raw_results):
        parsed = _parse_finbert_output(result)
        parsed["headline"] = headline
        parsed["ticker"] = ticker
        all_rows.append(parsed)

    details_df = pd.DataFrame(all_rows)

    # Group by ticker for per-headline detail DataFrames
    headline_details = {}
    if not details_df.empty:
        for ticker, group in details_df.groupby("ticker"):
            headline_details[ticker] = group.reset_index(drop=True)

    # Aggregate per ticker
    summary_rows = []
    for ticker in all_headlines:
        sector = tickers_sectors.get(ticker, "Unknown")
        company = TICKER_NAMES.get(ticker, ticker.replace(".NS", ""))

        if ticker in headline_details:
            hdf = headline_details[ticker]
            mean_net = hdf["net_score"].mean()
            pct_pos  = (hdf["label"] == "positive").mean() * 100
            pct_neg  = (hdf["label"] == "negative").mean() * 100
            pct_neu  = (hdf["label"] == "neutral").mean() * 100
            count    = len(hdf)
        else:
            mean_net = 0.0
            pct_pos = pct_neg = pct_neu = 0.0
            count = 0

        summary_rows.append({
            "ticker":         ticker,
            "company":        company,
            "sector":         sector,
            "mean_net_score": round(mean_net, 4),
            "pct_positive":   round(pct_pos, 1),
            "pct_negative":   round(pct_neg, 1),
            "pct_neutral":    round(pct_neu, 1),
            "article_count":  count,
        })

    ticker_summary = pd.DataFrame(summary_rows)
    ticker_summary = ticker_summary.sort_values(
        "mean_net_score", ascending=False
    ).reset_index(drop=True)

    return ticker_summary, headline_details


# =============================================================================
# 4. SECTOR-LEVEL AGGREGATION
# =============================================================================

def compute_sector_sentiment(ticker_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weighted average of constituent stocks' mean_net_score per sector.
    
    Returns DataFrame with columns: sector, mean_net_score, stock_count,
                                    pct_positive_avg, pct_negative_avg
    """
    if ticker_summary.empty:
        return pd.DataFrame(columns=["sector", "mean_net_score", "stock_count"])

    sector_agg = ticker_summary.groupby("sector").agg(
        mean_net_score=("mean_net_score", "mean"),
        stock_count=("ticker", "count"),
        pct_positive_avg=("pct_positive", "mean"),
        pct_negative_avg=("pct_negative", "mean"),
    ).reset_index()

    sector_agg["mean_net_score"] = sector_agg["mean_net_score"].round(4)
    sector_agg["pct_positive_avg"] = sector_agg["pct_positive_avg"].round(1)
    sector_agg["pct_negative_avg"] = sector_agg["pct_negative_avg"].round(1)

    return sector_agg.sort_values("mean_net_score", ascending=False).reset_index(drop=True)


# =============================================================================
# 5. MARKET BREADTH
# =============================================================================

def compute_market_breadth(ticker_summary: pd.DataFrame) -> float:
    """
    Percentage of NIFTY 50 stocks with positive net sentiment (mean_net_score > 0).
    """
    if ticker_summary.empty:
        return 0.0
    positive_count = (ticker_summary["mean_net_score"] > 0).sum()
    return round((positive_count / len(ticker_summary)) * 100, 1)


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick test with a few headlines
    pipe = load_finbert_pipeline()

    test_headlines = [
        "Reliance Industries reports record quarterly profit, beats estimates",
        "HDFC Bank faces regulatory scrutiny, shares decline 3%",
        "Infosys Q2 results in line with Street expectations",
    ]

    result_df = analyze_sentiment(test_headlines, pipe)
    print("\n--- Sentiment Analysis Results ---")
    print(result_df.to_string(index=False))
