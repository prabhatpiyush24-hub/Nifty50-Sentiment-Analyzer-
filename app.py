"""
app.py — Nifty 50 Sentiment Analyzer
=====================================================
High-performance Streamlit application providing:
  Tab 1: Market Overview (KPI cards, leaderboard, sector heatmap)
  Tab 2: Single-Stock Deep Dive (dual-axis chart, headline breakdown)
"""

import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from config import (
    NIFTY50_TICKERS, TICKER_NAMES, SECTORS, COLORS,
    APP_TITLE, APP_ICON, get_tickers_by_sector,
)
from data_fetcher import fetch_all_headlines, fetch_price_history
from model_engine import (
    load_finbert_pipeline, compute_ticker_sentiment,
    compute_sector_sentiment, compute_market_breadth,
)
from backtester import generate_backtest_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS — Premium dark theme with glassmorphism
# =============================================================================
st.markdown("""
<style>
    /* Import premium font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* KPI Metric Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30,30,46,0.9) 0%, rgba(40,40,60,0.7) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(124,77,255,0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(124,77,255,0.15);
    }
    .kpi-label {
        color: #A0A0B0;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 4px 0;
    }
    .kpi-sub {
        color: #A0A0B0;
        font-size: 0.8rem;
        font-weight: 400;
    }
    .positive { color: #00C853; }
    .negative { color: #FF1744; }
    .neutral  { color: #FFD600; }
    .accent   { color: #7C4DFF; }
    
    /* Help text styling */
    .help-text {
        color: #8888A0;
        font-size: 0.78rem;
        font-weight: 400;
        line-height: 1.5;
        margin: 4px 0 16px 0;
        padding: 10px 14px;
        background: rgba(124,77,255,0.05);
        border-left: 3px solid rgba(124,77,255,0.3);
        border-radius: 0 8px 8px 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(124,77,255,0.3);
    }
    
    /* Badge styles for sentiment labels */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-positive { background: rgba(0,200,83,0.15); color: #00C853; }
    .badge-negative { background: rgba(255,23,68,0.15); color: #FF1744; }
    .badge-neutral  { background: rgba(255,214,0,0.15); color: #FFD600; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1a1a2e 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED MODEL LOADER (singleton — loads once, persists across reruns)
# =============================================================================

@st.cache_resource(show_spinner="🔄 Loading FinBERT model...")
def get_model():
    """Load and cache the FinBERT pipeline (singleton)."""
    return load_finbert_pipeline()


# =============================================================================
# CACHED DATA PIPELINE (1-hour TTL, with progress indicators)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_data(_pipe):
    """
    Complete data pipeline: fetch headlines → run FinBERT → aggregate.
    Single cached function to avoid redundant work.
    Returns (ticker_summary, headline_details, sector_summary, market_breadth).
    """
    # Step 1: Parallel headline fetching (instant with mock, ~3s with RSS)
    headlines = fetch_all_headlines()

    # Step 2: Batch FinBERT inference
    ticker_summary, headline_details = compute_ticker_sentiment(headlines, _pipe)

    # Step 3: Aggregation
    sector_summary = compute_sector_sentiment(ticker_summary)
    breadth = compute_market_breadth(ticker_summary)

    return ticker_summary, headline_details, sector_summary, breadth


@st.cache_data(ttl=3600)
def get_last_refreshed_time():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def render_kpi_card(label: str, value: str, sub: str = "", color_class: str = "accent"):
    """Render a glassmorphism KPI card."""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {color_class}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def sentiment_color(score: float) -> str:
    """Return color based on sentiment score."""
    if score > 0.05:
        return COLORS["positive"]
    elif score < -0.05:
        return COLORS["negative"]
    return COLORS["neutral"]


def sentiment_class(score: float) -> str:
    """Return CSS class based on sentiment score."""
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    return "neutral"


def format_score(score: float) -> str:
    """Format score with sign and color indicator."""
    sign = "+" if score > 0 else ""
    return f"{sign}{score:.4f}"


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("# Nifty 50 Sentiment Analyzer")
    st.markdown("---")
    st.markdown("**Powered by:**")
    with st.popover("Tech Stack Explained"):
        st.markdown("""
        - **ProsusAI/FinBERT:** A pre-trained NLP model for financial sentiment.
        - **Plotly Interactive Charts:** For rendering interactive data visualizations.
        - **PyTorch Batch Inference:** Utilized for fast tensor computations and hardware-accelerated batch predictions.
        """)
    st.markdown("---")

    if st.button("Refresh All Data", width="stretch", type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div style="color: #A0A0B0; font-size: 0.75rem;">
    Headlines are fetched from RSS feeds with<br>
    fallback to curated mock data.<br><br>
    Sentiment scores: P(pos) - P(neg)<br>
    Range: [-1.0, +1.0]<br><br>
    Cache TTL: 1 hour<br>
    Last Refreshed: {get_last_refreshed_time()}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# LOAD DATA (single pipeline call)
# =============================================================================

pipe = get_model()
ticker_summary, headline_details, sector_summary, market_breadth = get_all_data(pipe)

# =============================================================================
# MAIN TABS
# =============================================================================

tab_overview, tab_deepdive, tab_validation = st.tabs(["📊 Market Overview", "🔬 Single-Stock Deep Dive", "✅ Model Validation"])


# =============================================================================
# TAB 1: MARKET OVERVIEW
# =============================================================================

with tab_overview:

    # --- KPI CARDS ROW ---
    st.markdown('<div class="section-header">🎯 NIFTY 50 Sentiment Pulse</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="help-text">These four cards summarize the overall market sentiment derived from recent financial news headlines. Each headline is scored by FinBERT (a finance-specific AI model) on a scale of <b>−1.0</b> (extremely negative) to <b>+1.0</b> (extremely positive). Scores near <b>0</b> indicate neutral sentiment.</div>', unsafe_allow_html=True)

    overall_score = ticker_summary["mean_net_score"].mean() if not ticker_summary.empty else 0.0
    # Guard against NaN (e.g. if all mean_net_scores are NaN)
    if pd.isna(overall_score):
        overall_score = 0.0
    top_gainer = ticker_summary.iloc[0] if not ticker_summary.empty else None
    top_decliner = ticker_summary.iloc[-1] if not ticker_summary.empty else None

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        score_class = sentiment_class(overall_score)
        render_kpi_card(
            "Overall Sentiment",
            format_score(overall_score),
            "Equal-weighted mean across all stocks",
            score_class,
        )

    with col2:
        breadth_class = "positive" if market_breadth > 50 else "negative"
        render_kpi_card(
            "Market Breadth",
            f"{market_breadth:.1f}%",
            "Stocks with positive net sentiment",
            breadth_class,
        )

    with col3:
        if top_gainer is not None:
            render_kpi_card(
                "🏆 Top Gainer",
                format_score(top_gainer["mean_net_score"]),
                f"{top_gainer['company']} ({top_gainer['ticker']})",
                "positive",
            )

    with col4:
        if top_decliner is not None:
            render_kpi_card(
                "📉 Top Decliner",
                format_score(top_decliner["mean_net_score"]),
                f"{top_decliner['company']} ({top_decliner['ticker']})",
                "negative",
            )

    with st.popover("ℹ️ How to read these cards"):
        st.markdown("""
        | Card | What it shows |
        |---|---|
        | **Overall Sentiment** | Average sentiment across all 50 Nifty 50 stocks. **Above +0.05 = Bullish**, **below −0.05 = Bearish**. |
        | **Market Breadth** | % of stocks with positive news sentiment. **Above 50% = majority bullish**, below 50% = majority bearish. |
        | **Top Gainer** | Stock with the highest (most positive) average sentiment score. |
        | **Top Decliner** | Stock with the lowest (most negative) average sentiment score. |
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TOP 5 & BOTTOM 5 MOVERS ---
    st.markdown('<div class="section-header">🚀 Top 5 & Bottom 5 Movers</div>', unsafe_allow_html=True)
    st.markdown('<div class="help-text">The <b>5 most positively</b> and <b>5 most negatively</b> perceived stocks based on their average news sentiment. Longer bars = stronger sentiment signal. Stocks appearing here consistently may indicate sustained market narratives.</div>', unsafe_allow_html=True)
    if not ticker_summary.empty:
        col_top, col_bot = st.columns(2)
        with col_top:
            top5 = ticker_summary.head(5)
            fig_top = px.bar(top5, x="mean_net_score", y="ticker", orientation="h", title="Top 5 Positive", color="mean_net_score", color_continuous_scale="Greens")
            fig_top.update_layout(template="plotly_dark", height=300, yaxis={'autorange':'reversed'}, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_top, width="stretch")
        with col_bot:
            bot5 = ticker_summary.tail(5)
            fig_bot = px.bar(bot5, x="mean_net_score", y="ticker", orientation="h", title="Top 5 Negative", color="mean_net_score", color_continuous_scale="Reds_r")
            fig_bot.update_layout(template="plotly_dark", height=300, yaxis={'autorange':'reversed'}, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_bot, width="stretch")

    # --- SENTIMENT DISTRIBUTION & RECENT NEWS ---
    st.markdown('<div class="section-header">🍩 Market Sentiment Distribution & Recent News</div>', unsafe_allow_html=True)
    st.markdown('<div class="help-text"><b>Left — Donut chart:</b> Shows the proportion of all analyzed headlines classified as positive (green), negative (red), or neutral (yellow). A balanced market typically has 30–40% neutral. <b>Right — Top news:</b> The strongest positive and negative headlines by net score across all stocks.</div>', unsafe_allow_html=True)
    col_dist, col_news = st.columns([1, 1])
    
    with col_dist:
        if headline_details:
            all_labels = pd.concat([df["label"] for df in headline_details.values()])
            counts = all_labels.value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            color_map = {"positive": COLORS["positive"], "negative": COLORS["negative"], "neutral": COLORS["neutral"]}
            fig_donut = px.pie(counts, names="Sentiment", values="Count", hole=0.5, color="Sentiment", color_discrete_map=color_map)
            fig_donut.update_layout(template="plotly_dark", height=350, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_donut, width="stretch")
        else:
            st.info("No sentiment distribution available.")
            
    with col_news:
        if headline_details:
            all_hl_df = pd.concat(headline_details.values())
            top_pos = all_hl_df[all_hl_df["label"] == "positive"].sort_values("net_score", ascending=False).head(3)
            top_neg = all_hl_df[all_hl_df["label"] == "negative"].sort_values("net_score", ascending=True).head(3)
            
            st.markdown("#### Top Positive News")
            for _, r in top_pos.iterrows():
                st.markdown(f"- **{r['ticker']}**: {r['headline']} (<span style='color:{COLORS['positive']}'>+{r['net_score']:.2f}</span>)", unsafe_allow_html=True)
                
            st.markdown("#### Top Negative News")
            for _, r in top_neg.iterrows():
                st.markdown(f"- **{r['ticker']}**: {r['headline']} (<span style='color:{COLORS['negative']}'>{r['net_score']:.2f}</span>)", unsafe_allow_html=True)
        else:
            st.info("No news available.")

    # --- LEADERBOARD (Full Width) ---
    st.markdown('<div class="section-header">📋 Cross-Sectional Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="help-text">Every Nifty 50 stock ranked by <b>Net Score</b> (= avg P(positive) − P(negative) across all headlines). Use the sector filter to narrow down. Key columns explained:</div>', unsafe_allow_html=True)
    with st.popover("ℹ️ Column definitions"):
        st.markdown("""
        | Column | Meaning |
        |---|---|
        | **Net Score** | Average sentiment: P(positive) − P(negative). Ranges from −1.0 (all negative) to +1.0 (all positive). |
        | **% Positive** | Percentage of this stock's headlines classified as positive by FinBERT. |
        | **% Negative** | Percentage of this stock's headlines classified as negative by FinBERT. |
        | **Articles** | Total number of news headlines analyzed for this stock. More articles = more reliable signal. |
        """)

    if not ticker_summary.empty:
        selected_sectors = st.multiselect("Filter by Sector", options=SECTORS, default=[])
        
        display_df = ticker_summary.copy()
        if selected_sectors:
            display_df = display_df[display_df["sector"].isin(selected_sectors)]
            
        display_df.insert(0, "Rank", range(1, len(display_df) + 1))
        display_df = display_df.rename(columns={
            "ticker":         "Ticker",
            "company":        "Company",
            "sector":         "Sector",
            "mean_net_score": "Net Score",
            "pct_positive":   "% Positive",
            "pct_negative":   "% Negative",
            "pct_neutral":    "% Neutral",
            "article_count":  "Articles",
        })

        st.dataframe(
            display_df[["Rank", "Ticker", "Company", "Sector",
                        "Net Score", "% Positive", "% Negative", "Articles"]],
            width="stretch",
            height=400,
            column_config={
                "Net Score": st.column_config.NumberColumn(format="%.4f"),
                "% Positive": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                "% Negative": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            },
        )
    else:
        st.warning("No sentiment data available.")

    # --- SECTOR HEATMAP & BUBBLE CHART ---
    st.markdown('<div class="section-header">🗺️ Sector Sentiment Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="help-text">Equal-weighted average of all constituent stocks\' sentiment within each sector. <b>Green bars</b> = sector has positive news flow. <b>Red bars</b> = sector faces negative narratives. The vertical zero-line separates bullish from bearish sectors.</div>', unsafe_allow_html=True)

    if not sector_summary.empty:
        fig_sector = go.Figure()

        colors = [sentiment_color(s) for s in sector_summary["mean_net_score"]]

        fig_sector.add_trace(go.Bar(
            x=sector_summary["mean_net_score"],
            y=sector_summary["sector"],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.1)", width=1),
            ),
            text=[f"{s:+.4f}" for s in sector_summary["mean_net_score"]],
            textposition="outside",
            textfont=dict(size=12, color="#FFFFFF"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Net Score: %{x:.4f}<br>"
                "<extra></extra>"
            ),
        ))

        fig_sector.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            margin=dict(l=10, r=80, t=10, b=10),
            xaxis=dict(
                title="Mean Net Sentiment Score",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.3)",
                gridcolor="rgba(255,255,255,0.05)",
            ),
            yaxis=dict(
                autorange="reversed",
                tickfont=dict(size=12),
            ),
            font=dict(family="Inter"),
        )

        st.plotly_chart(fig_sector, width="stretch")

        # Additional: Sector bubble chart
        st.markdown('<div class="section-header">🔵 Sector Sentiment vs. Coverage</div>', unsafe_allow_html=True)
        st.markdown('<div class="help-text">Each bubble represents a sector. <b>X-axis:</b> net sentiment score (further right = more positive). <b>Y-axis:</b> average % of positive headlines. <b>Bubble size:</b> number of stocks in that sector. Ideal sectors appear in the <b>top-right</b> quadrant (high positive %, high score).</div>', unsafe_allow_html=True)
        
        fig_bubble = go.Figure()
        fig_bubble.add_trace(go.Scatter(
            x=sector_summary["mean_net_score"],
            y=sector_summary["pct_positive_avg"],
            mode="markers+text",
            text=sector_summary["sector"],
            textposition="top center",
            textfont=dict(size=10, color="#FFFFFF"),
            marker=dict(
                size=sector_summary["stock_count"] * 12,
                color=sector_summary["mean_net_score"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Net Score", thickness=15),
                line=dict(width=1, color="rgba(255,255,255,0.2)"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Net Score: %{x:.4f}<br>"
                "Avg % Positive: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
        ))
        fig_bubble.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=550,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(
                title="Net Sentiment Score",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.3)",
                gridcolor="rgba(255,255,255,0.05)",
            ),
            yaxis=dict(
                title="Avg % Positive Headlines",
                gridcolor="rgba(255,255,255,0.05)",
            ),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_bubble, width="stretch")

    else:
        st.warning("No sector data available.")


# =============================================================================
# TAB 2: SINGLE-STOCK DEEP DIVE
# =============================================================================

with tab_deepdive:

    st.markdown('<div class="section-header">🔬 Single-Stock Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="help-text">Deep dive into a single stock. Select any Nifty 50 constituent to view its price action alongside AI-generated sentiment from recent news. This helps identify whether news flow aligns with price movement.</div>', unsafe_allow_html=True)

    # Stock selector
    ticker_options = sorted(NIFTY50_TICKERS.keys())
    ticker_labels = {t: f"{TICKER_NAMES.get(t, t)} ({t})" for t in ticker_options}

    col_sel, col_info = st.columns([2, 3])

    with col_sel:
        selected_ticker = st.selectbox(
            "Select Stock",
            options=ticker_options,
            format_func=lambda t: ticker_labels[t],
            index=ticker_options.index("RELIANCE.NS") if "RELIANCE.NS" in ticker_options else 0,
        )

    with col_info:
        if selected_ticker and not ticker_summary.empty:
            row = ticker_summary[ticker_summary["ticker"] == selected_ticker]
            if not row.empty:
                row = row.iloc[0]
                sc = sentiment_class(row["mean_net_score"])
                st.markdown(f"""
                <div class="kpi-card" style="padding: 16px;">
                    <span class="kpi-label">{row['company']}</span> · 
                    <span class="kpi-label">{row['sector']}</span> · 
                    <span class="kpi-value {sc}" style="font-size: 1.5rem;">
                        {format_score(row['mean_net_score'])}
                    </span> · 
                    <span class="kpi-sub">{row['article_count']} articles analyzed</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- DUAL-AXIS CHART: Price + Sentiment ---
    st.markdown('<div class="section-header">📈 Price × Sentiment Overlay (30 Days)</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="help-text"><b>Top panel:</b> 30-day closing price from NSE. <b>Bottom panel:</b> Daily average sentiment from news headlines — each bar represents one day\'s aggregated score. <b>Green bars</b> = positive news day, <b>red</b> = negative, <b>yellow</b> = neutral. The <b>dotted amber line</b> shows the 3-day rolling trend. Look for divergences — rising price with falling sentiment (or vice versa) can signal upcoming reversals.</div>', unsafe_allow_html=True)

    price_data = fetch_price_history(selected_ticker)

    if not price_data.empty and selected_ticker in headline_details:
        hd = headline_details[selected_ticker].copy()

        fig_dual = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.65, 0.35],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        )

        # --- Top panel: Price line ---
        fig_dual.add_trace(
            go.Scatter(
                x=price_data["Date"],
                y=price_data["Close"],
                name="Close Price (₹)",
                line=dict(color="#7C4DFF", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(124,77,255,0.06)",
                hovertemplate="₹%{y:,.2f}<extra>Close</extra>",
            ),
            row=1, col=1,
        )

        # --- Bottom panel: Daily aggregated sentiment ---
        # Aggregate headlines to trading-day level for clean visualization
        if "published" in hd.columns and hd["published"].notna().any():
            hd["pub_date"] = pd.to_datetime(hd["published"], errors="coerce").dt.normalize()
            hd = hd.dropna(subset=["pub_date"])

            daily_sent = hd.groupby("pub_date").agg(
                mean_score=("net_score", "mean"),
                count=("net_score", "count"),
                headlines=("headline", lambda x: " | ".join(x)),
            ).reset_index()
            daily_sent = daily_sent.sort_values("pub_date")
        else:
            # Fallback: spread evenly across price range
            n_hl = len(hd)
            if len(price_data) >= n_hl:
                idx = np.linspace(0, len(price_data) - 1, n_hl, dtype=int)
                dates = price_data["Date"].iloc[idx].values
            else:
                dates = price_data["Date"].values[:n_hl]
            daily_sent = pd.DataFrame({
                "pub_date": dates,
                "mean_score": hd["net_score"].values[:len(dates)],
                "count": 1,
                "headlines": hd["headline"].values[:len(dates)],
            })

        # Bar colors per day
        bar_colors = [
            COLORS["positive"] if s > 0.05
            else COLORS["negative"] if s < -0.05
            else COLORS["neutral"]
            for s in daily_sent["mean_score"]
        ]

        # Truncate headline text for hover (max 80 chars per headline, max 3)
        def _hover_text(row):
            hls = str(row["headlines"]).split(" | ")[:3]
            lines = [h[:80] + ("…" if len(h) > 80 else "") for h in hls]
            extra = row["count"] - len(lines)
            text = "<br>".join(lines)
            if extra > 0:
                text += f"<br>+{extra} more"
            return text

        hover_texts = daily_sent.apply(_hover_text, axis=1)

        # Sentiment bars
        fig_dual.add_trace(
            go.Bar(
                x=daily_sent["pub_date"],
                y=daily_sent["mean_score"],
                name="Daily Avg Sentiment",
                marker=dict(
                    color=bar_colors,
                    opacity=0.75,
                    line=dict(width=0.5, color="rgba(255,255,255,0.15)"),
                ),
                text=[f"{s:+.3f}" for s in daily_sent["mean_score"]],
                textposition="outside",
                textfont=dict(size=9, color="rgba(255,255,255,0.6)"),
                customdata=np.stack([daily_sent["count"].values, hover_texts.values], axis=-1),
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Avg Score: %{y:+.4f}<br>"
                    "Articles: %{customdata[0]}<br>"
                    "<br>%{customdata[1]}"
                    "<extra></extra>"
                ),
            ),
            row=2, col=1,
        )

        # Rolling 3-day sentiment trend line (if enough data points)
        if len(daily_sent) >= 3:
            window = min(3, len(daily_sent))
            daily_sent["trend"] = daily_sent["mean_score"].rolling(window=window, center=True, min_periods=1).mean()
            fig_dual.add_trace(
                go.Scatter(
                    x=daily_sent["pub_date"],
                    y=daily_sent["trend"],
                    name=f"{window}-Day Trend",
                    line=dict(color="#FFD600", width=2, dash="dot"),
                    hovertemplate="Trend: %{y:+.4f}<extra></extra>",
                ),
                row=2, col=1,
            )

        # Zero reference line on sentiment panel
        fig_dual.add_hline(
            y=0, row=2, col=1,
            line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"),
        )

        # Layout
        fig_dual.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=550,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11, color="#A0A0B0"),
                bgcolor="rgba(0,0,0,0)",
            ),
            font=dict(family="Inter"),
            hovermode="x unified",
            bargap=0.15,
        )

        # Y-axis styling
        fig_dual.update_yaxes(
            title_text="Close Price (₹)", row=1, col=1,
            gridcolor="rgba(255,255,255,0.05)",
            title_font=dict(size=11, color="#A0A0B0"),
        )
        fig_dual.update_yaxes(
            title_text="Sentiment Score", row=2, col=1,
            gridcolor="rgba(255,255,255,0.05)",
            range=[-1.05, 1.05],
            title_font=dict(size=11, color="#A0A0B0"),
            dtick=0.5,
        )

        # X-axis styling
        fig_dual.update_xaxes(gridcolor="rgba(255,255,255,0.05)", row=1, col=1)
        fig_dual.update_xaxes(gridcolor="rgba(255,255,255,0.05)", row=2, col=1)

        st.plotly_chart(fig_dual, use_container_width=True)

    elif price_data.empty:
        st.warning(f"No price data available for {selected_ticker}.")
    else:
        st.info("No headline details available for overlay.")

    # --- HEADLINE BREAKDOWN TABLE ---
    st.markdown('<div class="section-header">📰 Headline Breakdown</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="help-text">Every individual headline analyzed by FinBERT for this stock. The model assigns three probabilities: P(Pos), P(Neg), and P(Neu) — these always sum to 1.0. The <b>Net Score</b> = P(Pos) − P(Neg). <b>Confidence</b> shows how certain the model is about its top prediction.</div>', unsafe_allow_html=True)

    if selected_ticker in headline_details:
        hd = headline_details[selected_ticker].copy()

        display_headlines = hd[["published", "headline", "label", "confidence", "net_score",
                                 "p_positive", "p_negative", "p_neutral"]].copy()
        display_headlines = display_headlines.rename(columns={
            "published":   "Published",
            "headline":    "Headline",
            "label":       "Sentiment",
            "confidence":  "Confidence",
            "net_score":   "Net Score",
            "p_positive":  "P(Pos)",
            "p_negative":  "P(Neg)",
            "p_neutral":   "P(Neu)",
        })

        st.dataframe(
            display_headlines,
            width="stretch",
            height=400,
            column_config={
                "Published": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss", width="medium"),
                "Headline": st.column_config.TextColumn(width="large"),
                "Sentiment": st.column_config.TextColumn(width="small"),
                "Confidence": st.column_config.NumberColumn(format="%.4f"),
                "Net Score": st.column_config.NumberColumn(format="%.4f"),
                "P(Pos)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                "P(Neg)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                "P(Neu)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            },
        )

        # Quick stats
        st.markdown('<div class="help-text">Summary count of headlines by sentiment label. A healthy stock typically has a balanced mix. <b>Heavily skewed</b> distributions (e.g., 8 positive / 0 negative) may indicate a strong market narrative or potential bias in news sources.</div>', unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            pos_count = (hd["label"] == "positive").sum()
            st.metric("✅ Positive Headlines", pos_count, help="Number of headlines where FinBERT's top prediction was 'positive'.")
        with col_s2:
            neg_count = (hd["label"] == "negative").sum()
            st.metric("❌ Negative Headlines", neg_count, help="Number of headlines where FinBERT's top prediction was 'negative'.")
        with col_s3:
            neu_count = (hd["label"] == "neutral").sum()
            st.metric("⚪ Neutral Headlines", neu_count, help="Number of headlines where FinBERT's top prediction was 'neutral' — factual or non-committal news.")

    else:
        st.info(f"No headline details available for {selected_ticker}.")


# =============================================================================
# TAB 3: MODEL VALIDATION
# =============================================================================

with tab_validation:

    st.markdown('<div class="section-header">✅ Model Validation & Backtesting</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="help-text">This tab tests FinBERT\'s predictive power by comparing its <b>sentiment signals against actual next-day stock price returns</b>. For each stock, the model\'s daily sentiment is aligned with the following trading day\'s return. A directional accuracy <b>above 50%</b> means the model performs better than a coin flip. Statistical significance (p-value < 0.05) confirms the results aren\'t due to chance.</div>', unsafe_allow_html=True)

    # Run backtest (cached)
    @st.cache_data(ttl=3600, show_spinner="🔄 Running backtest against price data...")
    def get_backtest_report(_headline_details_keys, headline_details):
        return generate_backtest_report(headline_details)

    # Use ticker keys as cache key (dict is unhashable)
    report = get_backtest_report(
        tuple(sorted(headline_details.keys())),
        headline_details,
    )

    signals_df = report["signals_df"]
    accuracy = report["accuracy"]
    correlation = report["correlation"]
    confusion = report["confusion"]
    strategy = report["strategy"]

    # --- KPI CARDS ---
    st.markdown('<div class="section-header">🎯 Backtest Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="help-text"><b>Directional Accuracy:</b> % of times the sentiment correctly predicted whether the stock price would go up or down the next day. <b>Signal count:</b> total number of sentiment→return pairs tested. <b>Correlation:</b> statistical relationship between sentiment scores and returns (closer to ±1 = stronger).</div>', unsafe_allow_html=True)

    bc1, bc2, bc3, bc4 = st.columns(4)

    with bc1:
        acc = accuracy["overall_accuracy"]
        acc_class = "positive" if acc > 50 else "negative" if acc < 45 else "neutral"
        render_kpi_card("Directional Accuracy", f"{acc}%",
                        f"{accuracy['correct_signals']}/{accuracy['total_signals']} correct", acc_class)

    with bc2:
        render_kpi_card("Signals Tested", str(accuracy["total_signals"]),
                        f"{signals_df['ticker'].nunique() if not signals_df.empty else 0} stocks", "accent")

    with bc3:
        pr = correlation["pearson_r"]
        pr_class = "positive" if pr > 0.05 else "negative" if pr < -0.05 else "neutral"
        sig = "✓ Significant" if correlation["pearson_p"] < 0.05 else "✗ Not significant"
        render_kpi_card("Pearson Correlation", f"{pr:+.4f}", f"p={correlation['pearson_p']:.4f} ({sig})", pr_class)

    with bc4:
        sr = correlation["spearman_r"]
        sr_class = "positive" if sr > 0.05 else "negative" if sr < -0.05 else "neutral"
        sig_s = "✓ Significant" if correlation["spearman_p"] < 0.05 else "✗ Not significant"
        render_kpi_card("Spearman Correlation", f"{sr:+.4f}", f"p={correlation['spearman_p']:.4f} ({sig_s})", sr_class)

    with st.popover("ℹ️ How to interpret these metrics"):
        st.markdown("""
        | Metric | Good | Bad | Meaning |
        |---|---|---|---|
        | **Directional Accuracy** | >55% | <45% | % of correct up/down predictions |
        | **Pearson r** | >+0.10 | Near 0 | Linear relationship strength |
        | **Spearman r** | >+0.10 | Near 0 | Rank-order relationship strength |
        | **p-value** | <0.05 | >0.05 | Statistical significance (lower = more reliable) |
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- CONFUSION MATRIX ---
    st.markdown('<div class="section-header">📊 Confusion Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="help-text">Shows how the model\'s predicted sentiment direction (rows) maps to actual price movements (columns). <b>Diagonal cells</b> (top-left to bottom-right) represent correct predictions. Off-diagonal = errors. Ideal: high numbers on the diagonal, low numbers off it.</div>', unsafe_allow_html=True)

    if not confusion.empty:
        col_cm, col_cm_explain = st.columns([2, 1])
        with col_cm:
            # Heatmap
            fig_cm = go.Figure(data=go.Heatmap(
                z=confusion.values,
                x=confusion.columns.tolist(),
                y=confusion.index.tolist(),
                text=confusion.values,
                texttemplate="%{text}",
                textfont=dict(size=16, color="white"),
                colorscale=[
                    [0, "rgba(124,77,255,0.1)"],
                    [0.5, "rgba(124,77,255,0.4)"],
                    [1, "rgba(124,77,255,0.9)"],
                ],
                showscale=False,
                hovertemplate="%{y} → %{x}: %{z} signals<extra></extra>",
            ))
            fig_cm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                font=dict(family="Inter"),
                xaxis=dict(side="bottom"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_cm_explain:
            total = confusion.values.sum()
            if total > 0:
                # True predictions (diagonal)
                tp = confusion.iloc[0, 0]  # Predicted Positive, Actual Up
                tn = confusion.iloc[2, 2]  # Predicted Negative, Actual Down
                diag_sum = tp + tn
                st.markdown(f"""
                #### Key Numbers
                - **True Positive:** {tp} (predicted ↑, actually ↑)
                - **True Negative:** {tn} (predicted ↓, actually ↓)
                - **Correct calls:** {diag_sum}/{total} ({diag_sum/total*100:.0f}%)
                - **Neutral skips:** {confusion.iloc[1].sum()} (no trade signal)
                """)
            else:
                st.info("Not enough data for confusion matrix.")
    else:
        st.info("Not enough data for confusion matrix.")

    # --- STRATEGY VS BENCHMARK ---
    st.markdown('<div class="section-header">📈 Strategy vs Buy-and-Hold Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="help-text"><b>Sentiment Strategy:</b> Go long when daily avg sentiment > +0.05, go short when < −0.05, stay flat otherwise. <b>Benchmark:</b> Simple buy-and-hold (always long). If the <b>green line</b> (strategy) stays above the <b>gray line</b> (benchmark), the sentiment model adds value.</div>', unsafe_allow_html=True)

    if not strategy.empty and len(strategy) > 1:
        fig_strat = go.Figure()

        # Strategy line
        fig_strat.add_trace(go.Scatter(
            x=strategy["date"],
            y=strategy["strategy_cumulative_pct"],
            name="Sentiment Strategy",
            line=dict(color=COLORS["positive"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,200,83,0.06)",
            hovertemplate="Strategy: %{y:+.2f}%<extra></extra>",
        ))

        # Benchmark line
        fig_strat.add_trace(go.Scatter(
            x=strategy["date"],
            y=strategy["benchmark_cumulative_pct"],
            name="Buy & Hold Benchmark",
            line=dict(color="#A0A0B0", width=2, dash="dash"),
            hovertemplate="Benchmark: %{y:+.2f}%<extra></extra>",
        ))

        # Zero reference
        fig_strat.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"))

        fig_strat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=11, color="#A0A0B0"),
            ),
            yaxis=dict(
                title="Cumulative Return (%)",
                gridcolor="rgba(255,255,255,0.05)",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.15)",
            ),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            font=dict(family="Inter"),
            hovermode="x unified",
        )

        st.plotly_chart(fig_strat, use_container_width=True)

        # Strategy stats
        strat_ret = strategy["strategy_cumulative_pct"].iloc[-1] if len(strategy) > 0 else 0
        bench_ret = strategy["benchmark_cumulative_pct"].iloc[-1] if len(strategy) > 0 else 0
        alpha = strat_ret - bench_ret

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Strategy Return", f"{strat_ret:+.2f}%",
                      help="Total cumulative return of the sentiment-following strategy.")
        with sc2:
            st.metric("Benchmark Return", f"{bench_ret:+.2f}%",
                      help="Total cumulative return of a simple buy-and-hold approach.")
        with sc3:
            alpha_delta = f"{alpha:+.2f}%"
            st.metric("Alpha (Excess Return)", f"{alpha:+.2f}%", delta=alpha_delta,
                      delta_color="normal",
                      help="Strategy return minus benchmark. Positive = sentiment adds value.")
    else:
        st.info("Not enough aligned signal-return data to compute strategy returns.")

    # --- PER-SECTOR ACCURACY ---
    st.markdown('<div class="section-header">🏢 Per-Sector Accuracy</div>', unsafe_allow_html=True)
    st.markdown('<div class="help-text">Directional accuracy broken down by sector. Some sectors (e.g., IT, Banking) may be more predictable by news sentiment than others (e.g., Metals, Energy) which are driven more by commodity prices.</div>', unsafe_allow_html=True)

    if not accuracy["per_sector"].empty:
        sec_df = accuracy["per_sector"].copy()
        bar_colors_sec = [
            COLORS["positive"] if a > 50 else COLORS["negative"] if a < 45 else COLORS["neutral"]
            for a in sec_df["accuracy"]
        ]

        fig_sec = go.Figure(go.Bar(
            x=sec_df["accuracy"],
            y=sec_df["sector"],
            orientation="h",
            marker=dict(color=bar_colors_sec, line=dict(color="rgba(255,255,255,0.1)", width=1)),
            text=[f"{a:.0f}% ({s} signals)" for a, s in zip(sec_df["accuracy"], sec_df["signals"])],
            textposition="outside",
            textfont=dict(size=11, color="#FFFFFF"),
            hovertemplate="<b>%{y}</b><br>Accuracy: %{x:.1f}%<extra></extra>",
        ))

        # 50% reference line
        fig_sec.add_vline(x=50, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"))

        fig_sec.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=max(300, len(sec_df) * 45),
            margin=dict(l=10, r=100, t=10, b=10),
            xaxis=dict(
                title="Directional Accuracy (%)",
                range=[0, 105],
                gridcolor="rgba(255,255,255,0.05)",
            ),
            yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
            font=dict(family="Inter"),
        )

        st.plotly_chart(fig_sec, use_container_width=True)
    else:
        st.info("Not enough data for per-sector breakdown.")

    # --- PER-TICKER ACCURACY TABLE ---
    st.markdown('<div class="section-header">📋 Per-Stock Accuracy Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="help-text">Detailed accuracy for every individual stock. Stocks with <b>more signals</b> give more reliable accuracy estimates. Stocks with very few signals (1–2) should be interpreted with caution.</div>', unsafe_allow_html=True)

    if not accuracy["per_ticker"].empty:
        ticker_acc = accuracy["per_ticker"].copy()
        ticker_acc = ticker_acc.rename(columns={
            "ticker": "Ticker",
            "company": "Company",
            "sector": "Sector",
            "signals": "Signals",
            "correct": "Correct",
            "accuracy": "Accuracy (%)",
        })

        st.dataframe(
            ticker_acc[["Ticker", "Company", "Sector", "Signals", "Correct", "Accuracy (%)"]],
            use_container_width=True,
            height=400,
            column_config={
                "Accuracy (%)": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.1f%%",
                ),
                "Signals": st.column_config.NumberColumn(format="%d"),
                "Correct": st.column_config.NumberColumn(format="%d"),
            },
        )
    else:
        st.info("Not enough data for per-stock breakdown.")

    # --- RAW SIGNALS TABLE ---
    with st.expander("🔍 View raw signal-return data"):
        st.markdown('<div class="help-text">Every individual sentiment→return pair used in the backtest. Use this to audit specific predictions and understand model behavior on individual days.</div>', unsafe_allow_html=True)
        if not signals_df.empty:
            raw_display = signals_df.copy()
            raw_display["date"] = raw_display["date"].dt.strftime("%Y-%m-%d")
            raw_display = raw_display.rename(columns={
                "date": "Date",
                "ticker": "Ticker",
                "company": "Company",
                "sentiment_score": "Sentiment",
                "next_day_return": "Next-Day Return",
                "signal_correct": "Correct?",
                "headline_count": "Headlines",
            })
            st.dataframe(
                raw_display[["Date", "Ticker", "Company", "Sentiment",
                             "Next-Day Return", "Correct?", "Headlines"]],
                use_container_width=True,
                height=400,
                column_config={
                    "Sentiment": st.column_config.NumberColumn(format="%+.4f"),
                    "Next-Day Return": st.column_config.NumberColumn(format="%+.4f"),
                    "Correct?": st.column_config.CheckboxColumn(),
                },
            )
        else:
            st.info("No raw signal data available.")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #A0A0B0; font-size: 0.8rem; padding: 16px 0;">
    <strong>NIFTY 50 Quantitative Sentiment Dashboard</strong> · 
    Powered by ProsusAI/FinBERT · 
    Built with Streamlit & Plotly<br>
    <em>Disclaimer: This is a research tool. Not financial advice. Sentiment scores are model-generated and may not reflect actual market conditions.</em>
</div>
""", unsafe_allow_html=True)
