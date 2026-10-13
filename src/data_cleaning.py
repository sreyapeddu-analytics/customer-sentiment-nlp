"""
data_cleaning.py — text normalization helpers for the review dataset.

Steps: lowercase -> strip URLs -> keep letters/spaces -> collapse whitespace
-> optional stopword removal. Uses scikit-learn's built-in English stopword
list so no extra downloads are required.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
WS_RE = re.compile(r"\s+")


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Normalize a single review string."""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    if remove_stopwords:
        tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS]
        text = " ".join(tokens)
    return text


def add_sentiment_label(rating: int) -> str:
    """Derive a sentiment label from a 1-5 star rating."""
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


if __name__ == "__main__":
    sample = "LOVE this!! The sound-quality is AMAZING... visit https://x.co now!!"
    print("raw:     ", sample)
    print("cleaned: ", clean_text(sample))
    print("labels:  ", [(r, add_sentiment_label(r)) for r in (1, 2, 3, 4, 5)])
