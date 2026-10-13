"""
model.py — sentiment classification baselines on synthetic reviews.

Two scikit-learn pipelines on TF-IDF features:
    1. TfidfVectorizer(ngram_range=(1,2)) + LogisticRegression
    2. TfidfVectorizer(ngram_range=(1,2)) + MultinomialNB

Labels are derived from rating (1-2 negative, 3 neutral, 4-5 positive).
Stratified 80/20 train/test split, random_state=42.

Prints per-class precision/recall/F1, overall accuracy, the confusion matrix,
and the most positive/negative n-grams from the logistic regression weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data_cleaning import add_sentiment_label, clean_text

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "reviews.csv"
RANDOM_STATE = 42
LABELS = ["positive", "neutral", "negative"]


def load_xy() -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    df["sentiment"] = df["rating"].map(add_sentiment_label)
    df["clean_text"] = df["review_text"].map(clean_text)
    return df["clean_text"], df["sentiment"]


def build_pipelines() -> dict[str, Pipeline]:
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9)
    return {
        "LogisticRegression": Pipeline(
            [("tfidf", tfidf), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]
        ),
        "MultinomialNB": Pipeline(
            [("tfidf", tfidf), ("clf", MultinomialNB())]
        ),
    }


def top_ngrams(pipe: Pipeline, n: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strongest n-gram drivers per sentiment class.

    Uses the coefficient row of each target class directly: the highest
    weights in the 'positive' row predict positive reviews, the highest in
    the 'negative' row predict negative reviews.
    """
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    classes = list(clf.classes_)
    terms = np.array(vec.get_feature_names_out())

    def _top(class_name: str) -> pd.DataFrame:
        coefs = clf.coef_[classes.index(class_name)]
        order = np.argsort(coefs)[::-1]
        return pd.DataFrame(
            {"ngram": terms[order[:n]], "weight": coefs[order[:n]].round(3)}
        )

    return _top("positive"), _top("negative")


def main() -> None:
    X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  (stratified, seed=42)")

    pipes = build_pipelines()
    results = {}
    for name, pipe in pipes.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results[name] = (pipe, acc)
        print(f"\n{'=' * 60}\n{name} — accuracy: {acc:.4f}\n{'=' * 60}")
        print(classification_report(y_test, pred, labels=LABELS, digits=3))
        cm = confusion_matrix(y_test, pred, labels=LABELS)
        print("Confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

    best_name = max(results, key=lambda k: results[k][1])
    print(f"\nBest model: {best_name} (accuracy {results[best_name][1]:.4f})")

    lr_pipe = pipes["LogisticRegression"]
    pos, neg = top_ngrams(lr_pipe, n=15)
    print("\nTop positive n-grams (LogisticRegression):")
    print(pos.to_string(index=False))
    print("\nTop negative n-grams (LogisticRegression):")
    print(neg.to_string(index=False))


if __name__ == "__main__":
    main()
