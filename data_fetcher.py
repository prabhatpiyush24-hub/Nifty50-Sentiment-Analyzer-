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
    old_timeout = socket.getdefaulttimeout()

    try:
        # Set strict timeout for this thread
        socket.setdefaulttimeout(_RSS_TIMEOUT)

        # Try Yahoo Finance RSS
        try:
            yahoo_url = RSS_FEEDS["yahoo_finance"].format(ticker=ticker.replace(".NS", ""))
            feed = feedparser.parse(yahoo_url)
            if feed.bozo == 0 and feed.entries:
                for entry in feed.entries[:n]:
                    title = entry.get("title", "").strip()
                    if title and len(title) > 15:
                        headlines.append({"headline": title, "published": entry.get("published", entry.get("pubDate", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))})
        except Exception:
            pass

        # Try Google News RSS only if Yahoo didn't yield enough
        if len(headlines) < MIN_HEADLINES_PER_TICKER:
            try:
                google_url = RSS_FEEDS["google_news"].format(
                    company=company_name.replace(" ", "+")
                )
                feed = feedparser.parse(google_url)
                if feed.bozo == 0 and feed.entries:
                    for entry in feed.entries[:n]:
                        title = entry.get("title", "").strip()
                        if title and len(title) > 15 and not any(h.get("headline") == title for h in headlines):
                            headlines.append({"headline": title, "published": entry.get("published", entry.get("pubDate", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))})
            except Exception:
                pass

        if not headlines:
            with _rss_failures_lock:
                _rss_failures += 1
        else:
            with _rss_failures_lock:
                _rss_failures = 0  # Reset on success

    finally:
        socket.setdefaulttimeout(old_timeout)

    return headlines[:n]


# =============================================================================
# 2. FALLBACK MOCK HEADLINE GENERATOR
# =============================================================================

# Template pools for realistic financial headline generation
_POSITIVE_TEMPLATES = [
    "{company} reports record quarterly profit, beats Street estimates by 12%",
    "{company} shares surge 5% as Q{q} revenue grows 18% YoY",
    "Analysts upgrade {company} to 'Buy' citing strong order pipeline",
    "{company} announces ₹2,000 crore expansion plan, shares rally",
    "FIIs increase stake in {company} to 42%, signaling strong confidence",
    "{company} wins major contract worth ₹5,500 crore from government",
    "Morgan Stanley raises {company} target price to ₹{price}, maintains Overweight",
    "{company} declares interim dividend of ₹{div} per share",
    "{company} margin expansion surprises analysts; EBITDA up 320 bps",
    "Goldman Sachs adds {company} to Asia conviction list amid robust growth",
    "{company} launches new product line, expects 15% revenue boost",
    "{company} reports all-time high order book of ₹{order_book} crore",
    "Strong monsoon outlook lifts rural demand, {company} to benefit most",
    "{company} board approves share buyback program worth ₹1,200 crore",
    "{company} debt-free status achieved, balance sheet strongest in decade",
]

_NEGATIVE_TEMPLATES = [
    "{company} misses Q{q} earnings estimates; profit down 8% on weak demand",
    "{company} shares fall 4% after management cuts FY{fy} guidance",
    "SEBI investigation rattles {company} investors, stock drops 6%",
    "Analysts downgrade {company} to 'Sell' citing margin compression",
    "{company} faces supply chain disruption; warns of delayed deliveries",
    "Foreign investors dump {company} shares worth ₹3,200 crore in {month}",
    "Rising input costs squeeze {company} margins; EBITDA contracts 180 bps",
    "{company} loses key contract to competitor, market share under threat",
    "Credit Suisse downgrades {company}, slashes target price by 20%",
    "{company} CFO resignation raises corporate governance concerns",
    "{company} reports unexpected rise in NPAs, asset quality deteriorates",
    "Regulatory headwinds mount for {company} as government tightens norms",
    "{company} promoter pledge rises to 45%, raises red flags among investors",
    "Weak consumer spending hits {company} volume growth in Q{q}",
]

_NEUTRAL_TEMPLATES = [
    "{company} Q{q} results in line with estimates; no major surprises",
    "{company} board to meet on {date} to discuss routine agenda items",
    "{company} appoints new independent director as per regulatory norms",
    "{company} trading flat amid thin volumes and lack of fresh triggers",
    "{company} AGM scheduled for next month; routine resolutions on agenda",
    "{company} reports unchanged market share in latest industry data",
    "{company} maintains FY{fy} guidance without revision at analyst day",
    "{company} in early-stage talks with undisclosed party; no details yet",
    "Mutual fund holdings in {company} remain unchanged quarter-on-quarter",
    "{company} stock trades sideways as sector sees mixed institutional flows",
    "{company} annual report filed with exchanges; no material updates noted",
    "{company} management commentary offers no change in outlook for H2 FY{fy}",
]


def _generate_mock_headlines(ticker: str, company_name: str, n: int = 10) -> List[Dict[str, str]]:
    """
    Generate realistic, varied financial headlines for a given company.
    Mix of positive, negative, and neutral headlines with randomized parameters.
    This is instant (no I/O) and serves as the primary fast path.
    """
    quarters = [1, 2, 3, 4]
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    fy_years = [25, 26, 27]
    
    headlines = []
    
    # Determine sentiment distribution — balanced to avoid positive skew
    n_pos = random.randint(max(1, n // 4), max(2, n // 3))
    n_neg = random.randint(max(1, n // 4), max(2, n // 3))
    n_neu = n - n_pos - n_neg
    if n_neu < 0:
        # If we over-allocated pos+neg, trim whichever is larger
        excess = -n_neu
        if n_pos >= n_neg:
            n_pos -= excess
        else:
            n_neg -= excess
        n_neu = 0

    # Generate positive headlines
    pos_templates = random.sample(_POSITIVE_TEMPLATES, min(n_pos, len(_POSITIVE_TEMPLATES)))
    for tmpl in pos_templates:
        headline = tmpl.format(
            company=company_name,
            q=random.choice(quarters),
            fy=random.choice(fy_years),
            price=random.randint(500, 5000),
            div=random.choice([5, 8, 10, 12, 15, 20, 25]),
            order_book=random.randint(10000, 80000),
            month=random.choice(months),
            date=f"{random.randint(1,28)} {random.choice(months)}",
        )
        
        random_minutes = random.randint(1, 1440)
        pub_date = (datetime.datetime.now() - datetime.timedelta(minutes=random_minutes)).strftime("%Y-%m-%d %H:%M:%S")
        headlines.append({"headline": headline, "published": pub_date})

    # Generate negative headlines
    neg_templates = random.sample(_NEGATIVE_TEMPLATES, min(n_neg, len(_NEGATIVE_TEMPLATES)))
    for tmpl in neg_templates:
        headline = tmpl.format(
            company=company_name,
            q=random.choice(quarters),
            fy=random.choice(fy_years),
            price=random.randint(200, 3000),
            month=random.choice(months),
            date=f"{random.randint(1,28)} {random.choice(months)}",
        )
        
        random_minutes = random.randint(1, 1440)
        pub_date = (datetime.datetime.now() - datetime.timedelta(minutes=random_minutes)).strftime("%Y-%m-%d %H:%M:%S")
        headlines.append({"headline": headline, "published": pub_date})

    # Generate neutral headlines
    neu_templates = random.sample(_NEUTRAL_TEMPLATES, min(n_neu, len(_NEUTRAL_TEMPLATES)))
    for tmpl in neu_templates:
        headline = tmpl.format(
            company=company_name,
            q=random.choice(quarters),
            fy=random.choice(fy_years),
            date=f"{random.randint(1,28)} {random.choice(months)}",
        )
        
        random_minutes = random.randint(1, 1440)
        pub_date = (datetime.datetime.now() - datetime.timedelta(minutes=random_minutes)).strftime("%Y-%m-%d %H:%M:%S")
        headlines.append({"headline": headline, "published": pub_date})

    random.shuffle(headlines)
    return headlines[:n]


# =============================================================================
# 3. SINGLE-TICKER HEADLINE FETCHER
# =============================================================================

def fetch_headlines(ticker: str, company_name: str, n: int = MAX_HEADLINES_PER_TICKER) -> List[Dict[str, str]]:
    """
    Fetch headlines for a single ticker.
    Strategy: Try live RSS first (with 3s timeout) → fall back to mock generation.
    """
    # Attempt live RSS
    headlines = _fetch_rss_headlines(ticker, company_name, n)

    if len(headlines) >= MIN_HEADLINES_PER_TICKER:
        logger.info(f"[LIVE] {ticker}: fetched {len(headlines)} headlines from RSS")
        return headlines[:n]

    # Fall back to mock data (instant)
    mock = _generate_mock_headlines(ticker, company_name, n)
    if headlines:
        # Merge any partial RSS results with mock to reach target count
        combined = headlines + mock
        seen = set()
        deduped = []
        for h in combined:
            key = h["headline"]
            if key not in seen:
                seen.add(key)
                deduped.append(h)
        logger.info(f"[MIXED] {ticker}: {len(headlines)} live + {n - len(headlines)} mock headlines")
        return deduped[:n]

    logger.info(f"[MOCK] {ticker}: generated {len(mock)} mock headlines")
    return mock


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
