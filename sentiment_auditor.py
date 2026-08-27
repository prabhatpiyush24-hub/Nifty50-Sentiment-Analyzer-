import pandas as pd
import numpy as np
from typing import Dict, Any

def run_sentiment_audit(signals_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes pure NLP sentiment confidence and extracts edge cases.
    Returns a dictionary of metrics and specific headline DataFrames.
    """
    if signals_df.empty:
        return {}

    # Total distribution
    sentiment_counts = signals_df["sentiment"].value_counts().to_dict()
    total = len(signals_df)
    
    distribution = {
        "positive": sentiment_counts.get("positive", 0) / total * 100 if total > 0 else 0,
        "negative": sentiment_counts.get("negative", 0) / total * 100 if total > 0 else 0,
        "neutral": sentiment_counts.get("neutral", 0) / total * 100 if total > 0 else 0,
    }

    # High Confidence Extremes (Top 5 Positive, Top 5 Negative)
    # Sort by probability of the respective class
    pos_df = signals_df[signals_df["sentiment"] == "positive"].sort_values("pos_prob", ascending=False).head(5)
    neg_df = signals_df[signals_df["sentiment"] == "negative"].sort_values("neg_prob", ascending=False).head(5)

    # Confusing / Low Confidence (Where the max probability across classes is the lowest)
    signals_df["max_prob"] = signals_df[["pos_prob", "neg_prob", "neu_prob"]].max(axis=1)
    confusing_df = signals_df.sort_values("max_prob", ascending=True).head(10)

    # Convert to pure dicts for easy rendering
    return {
        "total_analyzed": total,
        "distribution": distribution,
        "high_confidence_positive": pos_df[["ticker", "headline", "pos_prob", "published"]].to_dict("records"),
        "high_confidence_negative": neg_df[["ticker", "headline", "neg_prob", "published"]].to_dict("records"),
        "confusing_headlines": confusing_df[["ticker", "headline", "pos_prob", "neg_prob", "neu_prob", "sentiment", "max_prob"]].to_dict("records"),
        "full_data": signals_df, # Keep full df for the explorer table
    }
