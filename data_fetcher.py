"""
data_fetcher.py — Data Ingestion Layer (Optimized for Speed)
==============================================================
Handles:
  1. RSS feed headline scraping with strict timeouts + parallel I/O
  2. Fallback realistic mock headline generation (instant)
  3. yfinance price history retrieval
  
Performance optimizations:
  - ThreadPoolExecutor for parallel headline fetching across 50 tickers
  - 3-second hard timeout on all RSS network calls
  - Fast-fail: if first 3 RSS attempts all fail, skip RSS entirely
  - Mock headlines generated instantly (no I/O)
"""

import random
import datetime
import logging
import socket
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import yfinance as yf

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

from config import (
    NIFTY50_TICKERS, TICKER_NAMES, RSS_FEEDS,
    MAX_HEADLINES_PER_TICKER, MIN_HEADLINES_PER_TICKER,
    PRICE_HISTORY_PERIOD,
)

logger = logging.getLogger(__name__)

# Set global socket timeout for RSS feeds (3 seconds max)
_RSS_TIMEOUT = 3

# Track RSS feed health — skip all feeds if first few fail
_rss_failures = 0
_rss_failures_lock = threading.Lock()
_RSS_FAILURE_THRESHOLD = 3  # After 3 consecutive failures, skip RSS entirely

# =============================================================================
# 1. RSS FEED HEADLINE FETCHING (with strict timeouts)
# =============================================================================

def _fetch_rss_headlines(ticker: str, company_name: str, n: int = 10) -> List[Dict[str, str]]:
    """
    Attempt to pull live headlines from RSS feeds with strict 3s timeout.
    Returns a list of headline strings, or empty list on failure.
    """
    global _rss_failures

    if not FEEDPARSER_AVAILABLE:
        return []

    # Fast-fail: skip RSS entirely if previous attempts consistently failed
    with _rss_failures_lock:
        if _rss_failures >= _RSS_FAILURE_THRESHOLD:
            return []

    headlines = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        # Try Yahoo Finance RSS
        try:
            yahoo_url = RSS_FEEDS["yahoo_finance"].format(ticker=ticker.replace(".NS", ""))
            r = requests.get(yahoo_url, headers=headers, timeout=_RSS_TIMEOUT)
            feed = feedparser.parse(r.text)
            if feed.entries:
                for entry in feed.entries[:n]:
                    title = entry.get("title", "").strip()
                    if title and len(title) > 15:
                        headlines.append({"headline": title, "published": entry.get("published", entry.get("pubDate", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))})
        except Exception as e:
            logger.debug(f"Yahoo fetch failed for {ticker}: {e}")

        # Try Google News RSS only if Yahoo didn't yield enough
        if len(headlines) < MIN_HEADLINES_PER_TICKER:
            try:
                google_url = RSS_FEEDS["google_news"].format(
                    company=company_name.replace(" ", "+")
                )
                r = requests.get(google_url, headers=headers, timeout=_RSS_TIMEOUT)
                feed = feedparser.parse(r.text)
                if feed.entries:
                    for entry in feed.entries[:n]:
                        title = entry.get("title", "").strip()
                        if title and len(title) > 15 and not any(h.get("headline") == title for h in headlines):
                            headlines.append({"headline": title, "published": entry.get("published", entry.get("pubDate", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))})
            except Exception as e:
                logger.debug(f"Google News fetch failed for {ticker}: {e}")

        if not headlines:
            with _rss_failures_lock:
                _rss_failures += 1
        else:
            with _rss_failures_lock:
                _rss_failures = 0  # Reset on success

    except Exception:
        pass

    return headlines[:n]


def fetch_headlines(ticker: str, company_name: str, n: int = MAX_HEADLINES_PER_TICKER) -> List[Dict[str, str]]:
    """
    Fetch headlines for a single ticker strictly using live RSS feeds.
    Returns up to n live headlines. No mock fallback.
    """
    headlines = _fetch_rss_headlines(ticker, company_name, n)
    logger.info(f"[LIVE] {ticker}: fetched {len(headlines)} headlines from RSS")
    return headlines[:n]


# =============================================================================
# 4. PARALLEL HEADLINE ORCHESTRATOR
# =============================================================================

def _fetch_single_ticker(args):
    """Worker function for parallel headline fetching."""
    ticker, company_name, n = args
    return ticker, fetch_headlines(ticker, company_name, n)


def fetch_all_headlines(
    tickers: Optional[Dict[str, str]] = None,
    progress_callback=None,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Fetch headlines for all tickers in the NIFTY 50 universe using parallel I/O.
    
    Args:
        tickers: Dict of {ticker: sector}. Defaults to NIFTY50_TICKERS.
        progress_callback: Optional callable(completed, total) for progress updates.
    
    Returns: {ticker: [headline_str, ...]}
    """
    if tickers is None:
        tickers = NIFTY50_TICKERS

    all_headlines = {}
    tasks = []
    for ticker in tickers:
        company_name = TICKER_NAMES.get(ticker, ticker.replace(".NS", ""))
        tasks.append((ticker, company_name, MAX_HEADLINES_PER_TICKER))

    total = len(tasks)
    completed = 0

    # Use ThreadPoolExecutor for parallel I/O (RSS feeds are I/O-bound)
    # max_workers=10 balances speed vs. rate-limiting
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_single_ticker, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                ticker, headlines = future.result(timeout=10)
                all_headlines[ticker] = headlines
            except Exception as e:
                # On any failure, generate mock data immediately
                task_args = futures[future]
                ticker = task_args[0]
                company_name = task_args[1]
                all_headlines[ticker] = _generate_mock_headlines(
                    ticker, company_name, MAX_HEADLINES_PER_TICKER
                )
                logger.warning(f"Parallel fetch failed for {ticker}: {e}")
            
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    total_headlines = sum(len(v) for v in all_headlines.values())
    logger.info(f"Total headlines fetched: {total_headlines} across {len(all_headlines)} tickers")
    return all_headlines


# =============================================================================
# 5. PRICE HISTORY
# =============================================================================

def fetch_price_history(
    ticker: str, period: str = PRICE_HISTORY_PERIOD
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a single ticker using yfinance.
    Returns a cleaned DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data.empty:
            logger.warning(f"No price data returned for {ticker}")
            return pd.DataFrame()
        
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data = data.reset_index()
        data["Date"] = pd.to_datetime(data["Date"])
        
        # Keep only the columns we need
        cols_to_keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in data.columns]
        data = data[cols_to_keep]
        
        return data.sort_values("Date").reset_index(drop=True)

    except Exception as e:
        logger.error(f"Price fetch failed for {ticker}: {e}")
        return pd.DataFrame()


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n--- Benchmarking parallel headline fetch for all NIFTY 50 ---")
    start = time.time()
    all_h = fetch_all_headlines()
    elapsed = time.time() - start
    total = sum(len(v) for v in all_h.values())
    print(f"Fetched {total} headlines for {len(all_h)} tickers in {elapsed:.2f}s")

    # Test price history
    print("\n--- Testing price history for TCS.NS ---")
    prices = fetch_price_history("TCS.NS")
    print(prices.tail())
