"""
config.py — NIFTY 50 Universe Configuration
=============================================
Central configuration for tickers, sectors, RSS feeds, model parameters,
and UI constants used across the Quantitative Sentiment Dashboard.
"""

# =============================================================================
# NIFTY 50 TICKERS → SECTOR MAPPING
# Each ticker uses the ".NS" suffix for Yahoo Finance (NSE) compatibility.
# =============================================================================
NIFTY50_TICKERS = {
    # --- IT ---
    "TCS.NS":           "IT",
    "INFY.NS":          "IT",
    "HCLTECH.NS":       "IT",
    "WIPRO.NS":         "IT",
    "TECHM.NS":         "IT",
    "LTIM.NS":          "IT",

    # --- Banking ---
    "HDFCBANK.NS":      "Banking",
    "ICICIBANK.NS":     "Banking",
    "KOTAKBANK.NS":     "Banking",
    "AXISBANK.NS":      "Banking",
    "SBIN.NS":          "Banking",
    "INDUSINDBK.NS":    "Banking",

    # --- Financial Services ---
    "BAJFINANCE.NS":    "Financial Services",
    "BAJAJFINSV.NS":    "Financial Services",
    "HDFC.NS":          "Financial Services",
    "SBILIFE.NS":       "Financial Services",

    # --- Energy ---
    "RELIANCE.NS":      "Energy",
    "ONGC.NS":          "Energy",
    "NTPC.NS":          "Energy",
    "POWERGRID.NS":     "Energy",
    "ADANIENT.NS":      "Energy",
    "BPCL.NS":          "Energy",
    "COALINDIA.NS":     "Energy",

    # --- Pharma ---
    "SUNPHARMA.NS":     "Pharma",
    "DRREDDY.NS":       "Pharma",
    "DIVISLAB.NS":      "Pharma",
    "CIPLA.NS":         "Pharma",
    "APOLLOHOSP.NS":    "Pharma",

    # --- Auto ---
    "TATAMOTORS.NS":    "Auto",
    "MARUTI.NS":        "Auto",
    "M&M.NS":           "Auto",
    "BAJAJ-AUTO.NS":    "Auto",
    "EICHERMOT.NS":     "Auto",
    "HEROMOTOCO.NS":    "Auto",

    # --- FMCG ---
    "HINDUNILVR.NS":    "FMCG",
    "ITC.NS":           "FMCG",
    "NESTLEIND.NS":     "FMCG",
    "BRITANNIA.NS":     "FMCG",
    "TATACONSUM.NS":    "FMCG",

    # --- Metals ---
    "TATASTEEL.NS":     "Metals",
    "JSWSTEEL.NS":      "Metals",
    "HINDALCO.NS":      "Metals",

    # --- Telecom ---
    "BHARTIARTL.NS":    "Telecom",

    # --- Infrastructure / Industrials ---
    "LT.NS":            "Infrastructure",
    "ULTRACEMCO.NS":    "Infrastructure",
    "GRASIM.NS":        "Infrastructure",
    "ADANIPORTS.NS":    "Infrastructure",

    # --- Diversified / Others ---
    "TITAN.NS":         "FMCG",
    "ASIANPAINT.NS":    "FMCG",
    "HDFCLIFE.NS":      "Financial Services",
    "UPL.NS":           "Pharma",
    "WIPRO.NS":         "IT",
}

# Human-readable company names for headline generation & display
TICKER_NAMES = {
    "TCS.NS":           "Tata Consultancy Services",
    "INFY.NS":          "Infosys",
    "HCLTECH.NS":       "HCL Technologies",
    "WIPRO.NS":         "Wipro",
    "TECHM.NS":         "Tech Mahindra",
    "LTIM.NS":          "LTIMindtree",
    "HDFCBANK.NS":      "HDFC Bank",
    "ICICIBANK.NS":     "ICICI Bank",
    "KOTAKBANK.NS":     "Kotak Mahindra Bank",
    "AXISBANK.NS":      "Axis Bank",
    "SBIN.NS":          "State Bank of India",
    "INDUSINDBK.NS":    "IndusInd Bank",
    "BAJFINANCE.NS":    "Bajaj Finance",
    "BAJAJFINSV.NS":    "Bajaj Finserv",
    "HDFC.NS":          "HDFC Ltd",
    "SBILIFE.NS":       "SBI Life Insurance",
    "RELIANCE.NS":      "Reliance Industries",
    "ONGC.NS":          "Oil & Natural Gas Corporation",
    "NTPC.NS":          "NTPC Ltd",
    "POWERGRID.NS":     "Power Grid Corporation",
    "ADANIENT.NS":      "Adani Enterprises",
    "BPCL.NS":          "Bharat Petroleum",
    "COALINDIA.NS":     "Coal India",
    "SUNPHARMA.NS":     "Sun Pharmaceutical",
    "DRREDDY.NS":       "Dr. Reddy's Laboratories",
    "DIVISLAB.NS":      "Divi's Laboratories",
    "CIPLA.NS":         "Cipla",
    "APOLLOHOSP.NS":    "Apollo Hospitals",
    "TATAMOTORS.NS":    "Tata Motors",
    "MARUTI.NS":        "Maruti Suzuki",
    "M&M.NS":           "Mahindra & Mahindra",
    "BAJAJ-AUTO.NS":    "Bajaj Auto",
    "EICHERMOT.NS":     "Eicher Motors",
    "HEROMOTOCO.NS":    "Hero MotoCorp",
    "HINDUNILVR.NS":    "Hindustan Unilever",
    "ITC.NS":           "ITC Ltd",
    "NESTLEIND.NS":     "Nestle India",
    "BRITANNIA.NS":     "Britannia Industries",
    "TATACONSUM.NS":    "Tata Consumer Products",
    "TATASTEEL.NS":     "Tata Steel",
    "JSWSTEEL.NS":      "JSW Steel",
    "HINDALCO.NS":      "Hindalco Industries",
    "BHARTIARTL.NS":    "Bharti Airtel",
    "LT.NS":            "Larsen & Toubro",
    "ULTRACEMCO.NS":    "UltraTech Cement",
    "GRASIM.NS":        "Grasim Industries",
    "ADANIPORTS.NS":    "Adani Ports",
    "TITAN.NS":         "Titan Company",
    "ASIANPAINT.NS":    "Asian Paints",
    "HDFCLIFE.NS":      "HDFC Life Insurance",
    "UPL.NS":           "UPL Ltd",
}

# =============================================================================
# SECTOR LISTING (unique sectors for grouping)
# =============================================================================
SECTORS = sorted(set(NIFTY50_TICKERS.values()))

def get_tickers_by_sector(sector: str) -> list:
    """Return list of tickers belonging to a given sector."""
    return [t for t, s in NIFTY50_TICKERS.items() if s == sector]

# =============================================================================
# RSS FEED SOURCES
# =============================================================================
RSS_FEEDS = {
    "yahoo_finance": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=IN&lang=en-IN",
    "google_news":   "https://news.google.com/rss/search?q={company}+stock+NSE&hl=en-IN&gl=IN&ceid=IN:en",
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 64
MAX_HEADLINES_PER_TICKER = 10
MIN_HEADLINES_PER_TICKER = 5

# =============================================================================
# UI CONSTANTS
# =============================================================================
APP_TITLE = "Nifty 50 Sentiment Analyzer"
APP_ICON = "📊"
PRICE_HISTORY_PERIOD = "1mo"

# Color palette for sentiment visualization
COLORS = {
    "positive":     "#00C853",   # Green
    "negative":     "#FF1744",   # Red
    "neutral":      "#FFD600",   # Amber
    "bg_dark":      "#0E1117",   # Dark background
    "card_bg":      "#1E1E2E",   # Card background
    "accent":       "#7C4DFF",   # Purple accent
    "text_primary": "#FFFFFF",
    "text_muted":   "#A0A0B0",
}
