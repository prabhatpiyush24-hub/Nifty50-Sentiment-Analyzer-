"""
backtester.py — Sentiment Model Validation & Backtesting Engine
================================================================
Compares FinBERT sentiment signals against actual next-day price returns
to measure the predictive value of the sentiment model.

Metrics computed:
  1. Directional accuracy (did sentiment predict price direction?)
  2. Sentiment–return correlation (Pearson & Spearman with p-values)
  3. Confusion matrix (TP/FP/TN/FN)
  4. Strategy vs benchmark cumulative returns
  5. Per-sector accuracy breakdown
"""

import logging
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
from scipy import stats

from config import NIFTY50_TICKERS, TICKER_NAMES
from data_fetcher import fetch_price_history

logger = logging.getLogger(__name__)


# =============================================================================
# 1. SIGNAL ALIGNMENT — Match daily sentiment with next-day returns
# =============================================================================

def _align_signals_for_ticker(
    ticker: str,
    headline_df: pd.DataFrame,
    price_period: str = "1mo",
) -> pd.DataFrame:
    """
    For a single ticker, align daily aggregated sentiment with next-day returns.

    Returns DataFrame with columns:
        date, ticker, sentiment_score, price_return, signal_correct
    """
    # Get price data
    price_data = fetch_price_history(ticker, period=price_period)
    if price_data.empty or len(price_data) < 3:
        return pd.DataFrame()

    # Daily returns
    price_data = price_data.sort_values("Date")
    price_data["return"] = price_data["Close"].pct_change()
    price_data["date"] = price_data["Date"].dt.normalize()

    # Aggregate headlines by day
    hd = headline_df.copy()
    if "published" not in hd.columns or hd["published"].isna().all():
        return pd.DataFrame()

    hd["date"] = pd.to_datetime(hd["published"], errors="coerce").dt.normalize()
    hd = hd.dropna(subset=["date"])

    daily_sentiment = hd.groupby("date").agg(
        sentiment_score=("net_score", "mean"),
        headline_count=("net_score", "count"),
    ).reset_index()

    # Merge: sentiment on day T → return on day T+1
    # Shift returns back by 1 day (so return[i] becomes the next-day return for date[i-1])
    price_returns = price_data[["date", "return"]].copy()
    price_returns["date"] = price_returns["date"] - pd.Timedelta(days=1)
    price_returns = price_returns.rename(columns={"return": "next_day_return"})

    # For weekends/holidays: expand the shift to find the closest prior trading day
    merged = pd.merge_asof(
        daily_sentiment.sort_values("date"),
        price_returns.sort_values("date"),
        on="date",
        direction="forward",
        tolerance=pd.Timedelta(days=4),  # Max 4-day gap (long weekends)
    )

    merged = merged.dropna(subset=["next_day_return", "sentiment_score"])

    if merged.empty:
        return pd.DataFrame()

    # Did sentiment predict direction correctly?
    merged["sentiment_direction"] = np.where(merged["sentiment_score"] > 0.05, 1,
                                    np.where(merged["sentiment_score"] < -0.05, -1, 0))
    merged["return_direction"] = np.where(merged["next_day_return"] > 0, 1,
                                 np.where(merged["next_day_return"] < 0, -1, 0))

    # Signal is correct when both agree on direction (skip neutral sentiment)
    merged["signal_correct"] = np.where(
        merged["sentiment_direction"] == 0, np.nan,  # Neutral = no signal
        (merged["sentiment_direction"] == merged["return_direction"]).astype(float)
    )

    merged["ticker"] = ticker
    merged["sector"] = NIFTY50_TICKERS.get(ticker, "Unknown")
    merged["company"] = TICKER_NAMES.get(ticker, ticker.replace(".NS", ""))

    return merged[["date", "ticker", "company", "sector", "sentiment_score",
                    "sentiment_direction", "next_day_return", "return_direction",
                    "signal_correct", "headline_count"]]


# =============================================================================
# 2. FULL BACKTEST — Run across all tickers
# =============================================================================

def run_backtest(
    headline_details: Dict[str, pd.DataFrame],
    price_period: str = "1mo",
) -> pd.DataFrame:
    """
    Run the backtest across all tickers with available headline data.

    Returns a combined DataFrame of aligned signals for all stocks.
    """
    all_signals = []

    for ticker, hdf in headline_details.items():
        try:
            signals = _align_signals_for_ticker(ticker, hdf, price_period)
            if not signals.empty:
                all_signals.append(signals)
        except Exception as e:
            logger.warning(f"Backtest failed for {ticker}: {e}")

    if not all_signals:
        return pd.DataFrame()

    combined = pd.concat(all_signals, ignore_index=True)
    logger.info(f"Backtest: {len(combined)} signal-return pairs across {combined['ticker'].nunique()} stocks")
    return combined


# =============================================================================
# 3. DIRECTIONAL ACCURACY
# =============================================================================

def compute_directional_accuracy(signals_df: pd.DataFrame) -> Dict:
    """
    Compute directional accuracy — how often sentiment predicted the right direction.

    Returns dict with:
        overall_accuracy, total_signals, correct_signals,
        per_ticker (DataFrame), per_sector (DataFrame)
    """
    if signals_df.empty:
        return {"overall_accuracy": 0.0, "total_signals": 0, "correct_signals": 0,
                "per_ticker": pd.DataFrame(), "per_sector": pd.DataFrame()}

    # Filter to non-neutral signals only (where model made a directional call)
    directional = signals_df.dropna(subset=["signal_correct"])

    if directional.empty:
        return {"overall_accuracy": 0.0, "total_signals": 0, "correct_signals": 0,
                "per_ticker": pd.DataFrame(), "per_sector": pd.DataFrame()}

    total = len(directional)
    correct = directional["signal_correct"].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0.0

    # Per-ticker accuracy
    per_ticker = directional.groupby(["ticker", "company", "sector"]).agg(
        signals=("signal_correct", "count"),
        correct=("signal_correct", "sum"),
    ).reset_index()
    per_ticker["accuracy"] = (per_ticker["correct"] / per_ticker["signals"] * 100).round(1)
    per_ticker = per_ticker.sort_values("accuracy", ascending=False)

    # Per-sector accuracy
    per_sector = directional.groupby("sector").agg(
        signals=("signal_correct", "count"),
        correct=("signal_correct", "sum"),
    ).reset_index()
    per_sector["accuracy"] = (per_sector["correct"] / per_sector["signals"] * 100).round(1)
    per_sector = per_sector.sort_values("accuracy", ascending=False)

    return {
        "overall_accuracy": round(accuracy, 1),
        "total_signals": int(total),
        "correct_signals": int(correct),
        "per_ticker": per_ticker,
        "per_sector": per_sector,
    }


# =============================================================================
# 4. CORRELATION ANALYSIS
# =============================================================================

def compute_correlation(signals_df: pd.DataFrame) -> Dict:
    """
    Compute Pearson and Spearman correlation between sentiment scores
    and next-day returns, with p-values.
    """
    if signals_df.empty or len(signals_df) < 5:
        return {
            "pearson_r": 0.0, "pearson_p": 1.0,
            "spearman_r": 0.0, "spearman_p": 1.0,
            "n_samples": 0,
        }

    clean = signals_df.dropna(subset=["sentiment_score", "next_day_return"])

    if len(clean) < 5:
        return {
            "pearson_r": 0.0, "pearson_p": 1.0,
            "spearman_r": 0.0, "spearman_p": 1.0,
            "n_samples": 0,
        }

    pearson_r, pearson_p = stats.pearsonr(clean["sentiment_score"], clean["next_day_return"])
    spearman_r, spearman_p = stats.spearmanr(clean["sentiment_score"], clean["next_day_return"])

    return {
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 4),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 4),
        "n_samples": len(clean),
    }


# =============================================================================
# 5. CONFUSION MATRIX
# =============================================================================

def compute_confusion_matrix(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute confusion matrix: sentiment direction vs actual return direction.

    Returns a 3x3 DataFrame (predicted vs actual) with labels:
        Positive, Neutral, Negative
    """
    if signals_df.empty:
        return pd.DataFrame(
            np.zeros((3, 3), dtype=int),
            index=["Predicted Positive", "Predicted Neutral", "Predicted Negative"],
            columns=["Actual Up", "Actual Flat", "Actual Down"],
        )

    df = signals_df.copy()

    # Map directions to labels
    pred_map = {1: "Predicted Positive", 0: "Predicted Neutral", -1: "Predicted Negative"}
    actual_map = {1: "Actual Up", 0: "Actual Flat", -1: "Actual Down"}

    df["pred_label"] = df["sentiment_direction"].map(pred_map)
    df["actual_label"] = df["return_direction"].map(actual_map)

    # Cross-tabulate
    ct = pd.crosstab(
        df["pred_label"], df["actual_label"],
        rownames=["Predicted"], colnames=["Actual"],
    )

    # Ensure all rows/columns exist
    for label in ["Predicted Positive", "Predicted Neutral", "Predicted Negative"]:
        if label not in ct.index:
            ct.loc[label] = 0
    for label in ["Actual Up", "Actual Flat", "Actual Down"]:
        if label not in ct.columns:
            ct[label] = 0

    ct = ct.reindex(
        index=["Predicted Positive", "Predicted Neutral", "Predicted Negative"],
        columns=["Actual Up", "Actual Flat", "Actual Down"],
    ).fillna(0).astype(int)

    return ct


# =============================================================================
# 6. STRATEGY RETURNS — Sentiment-following vs Buy-and-Hold
# =============================================================================

def compute_strategy_returns(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare a simple sentiment-following strategy against buy-and-hold.

    Strategy rules:
        - If daily avg sentiment > +0.05: go long (weight = +1)
        - If daily avg sentiment < -0.05: go short (weight = -1)
        - Otherwise: stay flat (weight = 0)

    Returns DataFrame with columns: date, strategy_return, benchmark_return,
                                     strategy_cumulative, benchmark_cumulative
    """
    if signals_df.empty:
        return pd.DataFrame()

    # Aggregate across all stocks per day
    daily = signals_df.groupby("date").agg(
        avg_sentiment=("sentiment_score", "mean"),
        avg_return=("next_day_return", "mean"),
    ).reset_index().sort_values("date")

    if daily.empty:
        return pd.DataFrame()

    # Strategy signal
    daily["weight"] = np.where(daily["avg_sentiment"] > 0.05, 1.0,
                      np.where(daily["avg_sentiment"] < -0.05, -1.0, 0.0))

    # Strategy return = weight × next-day return
    daily["strategy_return"] = daily["weight"] * daily["avg_return"]
    daily["benchmark_return"] = daily["avg_return"]

    # Cumulative returns (compound)
    daily["strategy_cumulative"] = (1 + daily["strategy_return"]).cumprod() - 1
    daily["benchmark_cumulative"] = (1 + daily["benchmark_return"]).cumprod() - 1

    # Convert to percentage
    daily["strategy_cumulative_pct"] = daily["strategy_cumulative"] * 100
    daily["benchmark_cumulative_pct"] = daily["benchmark_cumulative"] * 100

    return daily[["date", "avg_sentiment", "weight", "strategy_return", "benchmark_return",
                   "strategy_cumulative", "benchmark_cumulative",
                   "strategy_cumulative_pct", "benchmark_cumulative_pct"]]


# =============================================================================
# 7. SUMMARY REPORT — One-call convenience function
# =============================================================================

def generate_backtest_report(
    headline_details: Dict[str, pd.DataFrame],
    price_period: str = "1mo",
) -> Dict:
    """
    Run the full backtest and return all metrics in a single dict.

    Returns:
        signals_df: Raw aligned signals DataFrame
        accuracy: Directional accuracy metrics
        correlation: Correlation metrics
        confusion: Confusion matrix DataFrame
        strategy: Strategy returns DataFrame
    """
    signals_df = run_backtest(headline_details, price_period)

    return {
        "signals_df": signals_df,
        "accuracy": compute_directional_accuracy(signals_df),
        "correlation": compute_correlation(signals_df),
        "confusion": compute_confusion_matrix(signals_df),
        "strategy": compute_strategy_returns(signals_df),
    }


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick test with mock data
    from data_fetcher import fetch_all_headlines
    from model_engine import load_finbert_pipeline, compute_ticker_sentiment

    pipe = load_finbert_pipeline()
    headlines = fetch_all_headlines()
    ticker_summary, headline_details = compute_ticker_sentiment(headlines, pipe)

    report = generate_backtest_report(headline_details)

    print(f"\n--- Backtest Report ---")
    print(f"Signal-return pairs: {len(report['signals_df'])}")
    print(f"Directional accuracy: {report['accuracy']['overall_accuracy']}%")
    print(f"Pearson r: {report['correlation']['pearson_r']} (p={report['correlation']['pearson_p']})")
    print(f"\nConfusion Matrix:")
    print(report['confusion'])
    print(f"\nStrategy Returns:")
    print(report['strategy'].tail())
