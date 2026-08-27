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

tab_overview, tab_deepdive = st.tabs(["📊 Market Overview", "🔬 Single-Stock Deep Dive"])


# =============================================================================
# TAB 1: MARKET OVERVIEW
# =============================================================================

with tab_overview:

    # --- KPI CARDS ROW ---
    st.markdown('<div class="section-header">🎯 NIFTY 50 Sentiment Pulse</div>',
                unsafe_allow_html=True)

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

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TOP 5 & BOTTOM 5 MOVERS ---
    st.markdown('<div class="section-header">🚀 Top 5 & Bottom 5 Movers</div>', unsafe_allow_html=True)
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

    price_data = fetch_price_history(selected_ticker)

    if not price_data.empty and selected_ticker in headline_details:
        hd = headline_details[selected_ticker]

        fig_dual = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]],
        )

        # Price line (primary y-axis)
        fig_dual.add_trace(
            go.Scatter(
                x=price_data["Date"],
                y=price_data["Close"],
                name="Close Price (₹)",
                line=dict(color="#7C4DFF", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(124,77,255,0.08)",
                hovertemplate="₹%{y:,.2f}<extra>Close Price</extra>",
            ),
            secondary_y=False,
        )

        # Sentiment scores as bar overlay (secondary y-axis)
        # Use actual published dates from headlines
        hd_sorted = hd.copy()
        if "published" in hd_sorted.columns and hd_sorted["published"].notna().any():
            hd_sorted = hd_sorted.dropna(subset=["published"]).sort_values("published")
            sentiment_dates = hd_sorted["published"].values
        else:
            # Fallback: distribute evenly if published dates are missing
            n_headlines = len(hd)
            if len(price_data) >= n_headlines:
                date_indices = np.linspace(0, len(price_data) - 1, n_headlines, dtype=int)
                sentiment_dates = price_data["Date"].iloc[date_indices].values
            else:
                sentiment_dates = price_data["Date"].values[:n_headlines]
            hd_sorted = hd.iloc[:len(sentiment_dates)]

        bar_colors = [
            COLORS["positive"] if s > 0.05
            else COLORS["negative"] if s < -0.05
            else COLORS["neutral"]
            for s in hd_sorted["net_score"].values
        ]

        fig_dual.add_trace(
            go.Bar(
                x=sentiment_dates,
                y=hd_sorted["net_score"].values,
                name="Headline Sentiment",
                marker=dict(
                    color=bar_colors,
                    opacity=0.6,
                    line=dict(width=0),
                ),
                hovertemplate="Score: %{y:.4f}<extra>Sentiment</extra>",
            ),
            secondary_y=True,
        )

        fig_dual.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11),
            ),
            font=dict(family="Inter"),
            hovermode="x unified",
        )

        fig_dual.update_yaxes(
            title_text="Close Price (₹)",
            secondary_y=False,
            gridcolor="rgba(255,255,255,0.05)",
        )
        fig_dual.update_yaxes(
            title_text="Sentiment Score",
            secondary_y=True,
            gridcolor="rgba(255,255,255,0.03)",
            range=[-1, 1],
        )
        fig_dual.update_xaxes(gridcolor="rgba(255,255,255,0.05)")

        st.plotly_chart(fig_dual, width="stretch")

    elif price_data.empty:
        st.warning(f"No price data available for {selected_ticker}.")
    else:
        st.info("No headline details available for overlay.")

    # --- HEADLINE BREAKDOWN TABLE ---
    st.markdown('<div class="section-header">📰 Headline Breakdown</div>',
                unsafe_allow_html=True)

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
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            pos_count = (hd["label"] == "positive").sum()
            st.metric("✅ Positive Headlines", pos_count)
        with col_s2:
            neg_count = (hd["label"] == "negative").sum()
            st.metric("❌ Negative Headlines", neg_count)
        with col_s3:
            neu_count = (hd["label"] == "neutral").sum()
            st.metric("⚪ Neutral Headlines", neu_count)

    else:
        st.info(f"No headline details available for {selected_ticker}.")


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
