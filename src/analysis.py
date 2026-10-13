"""
analysis.py — exploratory data analysis helpers.

Prints: rating distribution, sentiment share by product category, and review
length vs rating. Sentiment is derived from rating (1-2 negative, 3 neutral,
4-5 positive).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_cleaning import add_sentiment_label, clean_text

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "reviews.csv"


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["sentiment"] = df["rating"].map(add_sentiment_label)
    df["clean_text"] = df["review_text"].map(lambda t: clean_text(t, remove_stopwords=False))
    df["word_count"] = df["clean_text"].str.split().str.len()
    return df


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df["rating"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .rename("pct")
        .reset_index()
        .sort_values("rating")
    )
    return out


def sentiment_share_by_category(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["product_category", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )
    pct = out.div(out.sum(axis=1), axis=0).mul(100).round(1)
    return pct[["positive", "neutral", "negative"]]


def length_vs_rating(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("rating")["word_count"]
        .agg(["mean", "median", "count"])
        .round(1)
        .rename(columns={"mean": "avg_words", "median": "median_words", "count": "n_reviews"})
    )


def main() -> None:
    df = load()
    print(f"Loaded {len(df):,} reviews")
    print("\n== Rating distribution (%) ==")
    print(rating_distribution(df).to_string(index=False))
    print("\n== Sentiment share by category (%) ==")
    print(sentiment_share_by_category(df).to_string())
    print("\n== Review length vs rating ==")
    print(length_vs_rating(df).to_string())
    print("\n== Overall sentiment ==")
    print(df["sentiment"].value_counts(normalize=True).mul(100).round(1).to_string())


if __name__ == "__main__":
    main()
