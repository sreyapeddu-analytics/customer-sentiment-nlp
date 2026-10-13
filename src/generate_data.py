"""
generate_data.py — create a synthetic customer-review dataset for portfolio NLP work.

All reviews are programmatically generated (no real customers, no real products or
brands). A fixed seed (random_state=42) makes the dataset reproducible.

Columns:
    review_id         unique integer id
    review_date       date of the (fictional) review
    product_category  one of 5 generic product categories
    rating            integer 1-5
    review_text       synthetic review text

Sentiment labels are NOT stored as a column; they are derived from rating in
analysis/model code (1-2 = negative, 3 = neutral, 4-5 = positive).
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

SEED = 42
N_REVIEWS = 5000

CATEGORIES = {
    "headphones": {
        "features": ["sound quality", "bass", "noise cancellation", "battery life",
                     "ear cushions", "bluetooth connection", "microphone", "fit"],
    },
    "coffee maker": {
        "features": ["brew strength", "water reservoir", "carafe", "warming plate",
                     "grinder", "drip tray", "timer", "filter basket"],
    },
    "running shoes": {
        "features": ["cushioning", "arch support", "tread", "laces", "toe box",
                     "heel grip", "breathability", "sole"],
    },
    "blender": {
        "features": ["motor", "blades", "jar", "speed settings", "lid",
                     "pulse function", "base", "cord"],
    },
    "smartwatch": {
        "features": ["screen", "battery life", "heart rate sensor", "gps",
                     "watch band", "touchscreen", "app sync", "sleep tracking"],
    },
}

POSITIVE_WORDS = [
    "excellent", "amazing", "fantastic", "wonderful", "outstanding", "superb",
    "impressive", "reliable", "durable", "comfortable", "fast", "easy",
    "convenient", "powerful", "sleek", "sturdy", "crisp", "clear", "vibrant",
    "bright", "responsive", "quiet", "lightweight", "compact", "versatile",
    "efficient", "intuitive", "smooth", "flawless", "awesome", "brilliant",
    "solid", "premium", "sharp", "accurate", "seamless",
]
NEGATIVE_WORDS = [
    "terrible", "awful", "horrible", "poor", "disappointing", "broken",
    "defective", "cheap", "flimsy", "uncomfortable", "slow", "difficult",
    "complicated", "weak", "noisy", "bulky", "heavy", "useless", "damaged",
    "scratched", "unreliable", "confusing", "frustrating", "missing", "wrong",
    "laggy", "blurry", "dim", "unresponsive", "leaking", "cracked",
    "overheating", "dull", "scratchy", "wobbly", "sticky",
]
NEUTRAL_WORDS = [
    "okay", "average", "decent", "fine", "acceptable", "standard", "typical",
    "ordinary", "moderate", "basic", "reasonable", "mediocre", "middling",
]

POSITIVE_VERBS = ["love", "adore", "recommend", "enjoy", "appreciate", "praise"]
NEGATIVE_VERBS = ["hate", "regret", "dislike", "distrust", "resent"]

POSITIVE_TEMPLATES = [
    "I {verb} this {category}! The {feature} is {adj} and it works {adj2}.",
    "Honestly the {feature} is {adj}. Setup was {adj2} and I use it every day.",
    "Five stars for the {feature} alone - {adj}, {adj2}, and totally worth it.",
    "The {feature} exceeded my expectations: {adj} performance and {adj2} build.",
    "Bought this for the {feature} and it is {adj}. Would {verb} it to anyone.",
    "So {adj}! The {feature} feels {adj2} and everything just works.",
    "Really {adj} {category}. The {feature} is {adj2} - money well spent.",
    "The {feature} is hands down {adj}. Packaging, delivery, quality: all {adj2}.",
    "After a month, the {feature} is still {adj}. Could not be happier with this {category}.",
    "Super {adj} and the {feature} is {adj2}. I {verb} buying this.",
]

NEGATIVE_TEMPLATES = [
    "I {verb} this {category}. The {feature} is {adj} and it broke within a week.",
    "Really {adj}. The {feature} feels {adj2} - requesting a refund.",
    "The {feature} is {adj} and customer service was {adj2}. Avoid.",
    "Stopped working after two days. The {feature} is completely {adj}.",
    "What a {adj} purchase. The {feature} arrived {adj2} and half the parts were {adj3}.",
    "The {feature} is {adj}, the {feature2} is {adj2}. Total waste of money.",
    "Do not buy. {feature_cap} is {adj} and the whole {category} feels {adj2}.",
    "One star. The {feature} is {adj} and assembly was {adj2}.",
    "The {feature} stopped working - {adj} quality and {adj2} design.",
    "Returned it. The {feature} was {adj} and nothing about this {category} is {adj2}.",
]

NEUTRAL_TEMPLATES = [
    "It is {adj}. The {feature} is {adj2}, nothing special but it works.",
    "The {feature} is {adj} for the price. Does the job, {adj2} overall.",
    "{adj_cap} {category}. The {feature} could be {adj2}, but I am not complaining.",
    "Works as {adj}. The {feature} is {adj2} - fine for casual use.",
    "The {feature} is {adj}. Shipping was {adj2}. Not great, not terrible.",
    "It is a {adj} {category}. The {feature} performs {adj2} as expected.",
]

MIXED_TEMPLATES = [  # 3-star reviews that mix mild praise with mild criticism
    "The {feature} is {pos}, but the {feature2} is {neg}. Mixed feelings overall.",
    "{pos_cap} build on the {feature}, however the {feature2} feels {neg}.",
    "I like the {feature} ({pos}), but the {feature2} is honestly {neg}.",
]

# Complaint follow-ups: unhappy customers explain themselves, so negative
# reviews get 1-2 extra sentences (longer on average than praise).
NEGATIVE_FOLLOWUPS = [
    "The {feature} started acting {neg} after just a few days of use.",
    "I contacted support and they were completely {neg}.",
    "Save your money and look elsewhere.",
    "The {feature2} is also {neg}, which somehow makes it even worse.",
    "Even the packaging arrived {neg}.",
    "For this price I expected far better than {neg} quality.",
]


def _pick(rng: random.Random, pool: list[str]) -> str:
    return rng.choice(pool)


def _generate_review(rng: random.Random, category: str, rating: int) -> str:
    features = CATEGORIES[category]["features"]
    if rating >= 4:
        t = rng.choice(POSITIVE_TEMPLATES)
        return t.format(
            verb=_pick(rng, POSITIVE_VERBS),
            category=category,
            feature=rng.choice(features),
            adj=_pick(rng, POSITIVE_WORDS),
            adj2=_pick(rng, POSITIVE_WORDS),
        )
    if rating <= 2:
        t = rng.choice(NEGATIVE_TEMPLATES)
        text = t.format(
            verb=_pick(rng, NEGATIVE_VERBS),
            category=category,
            feature=rng.choice(features),
            feature2=rng.choice(features),
            feature_cap=rng.choice(features).capitalize(),
            adj=_pick(rng, NEGATIVE_WORDS),
            adj2=_pick(rng, NEGATIVE_WORDS),
            adj3=_pick(rng, NEGATIVE_WORDS),
        )
        # Complaints run long: append 1-2 follow-up sentences.
        for _ in range(rng.choice([1, 1, 2])):
            text += " " + rng.choice(NEGATIVE_FOLLOWUPS).format(
                feature=rng.choice(features),
                feature2=rng.choice(features),
                neg=_pick(rng, NEGATIVE_WORDS),
            )
        return text
    # rating == 3: neutral or mildly mixed
    if rng.random() < 0.35:
        t = rng.choice(MIXED_TEMPLATES)
        return t.format(
            feature=rng.choice(features),
            feature2=rng.choice(features),
            pos=_pick(rng, POSITIVE_WORDS),
            pos_cap=_pick(rng, POSITIVE_WORDS).capitalize(),
            neg=_pick(rng, NEGATIVE_WORDS),
        )
    t = rng.choice(NEUTRAL_TEMPLATES)
    return t.format(
        category=category,
        feature=rng.choice(features),
        adj=_pick(rng, NEUTRAL_WORDS),
        adj_cap=_pick(rng, NEUTRAL_WORDS).capitalize(),
        adj2=_pick(rng, NEUTRAL_WORDS),
    )


def main() -> None:
    rng = random.Random(SEED)
    # Typical e-commerce skew: most reviews are 4-5 stars
    rating_weights = {1: 0.10, 2: 0.08, 3: 0.15, 4: 0.30, 5: 0.37}
    ratings = list(rating_weights)
    weights = list(rating_weights.values())
    categories = list(CATEGORIES)

    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2025-12-31")
    span_days = (end - start).days

    rows = []
    for i in range(1, N_REVIEWS + 1):
        rating = rng.choices(ratings, weights=weights, k=1)[0]
        category = rng.choice(categories)
        # 8% label noise: real review text sometimes disagrees with the star
        # rating (sarcasm, mixed feelings, misclicks). Keeps the task honest.
        template_rating = rating
        if rng.random() < 0.08:
            if rating >= 4:
                template_rating = 2
            elif rating <= 2:
                template_rating = 5
            else:
                template_rating = rng.choice([2, 4])
        text = _generate_review(rng, category, template_rating)
        # Longer reviews skew slightly negative (complaints run long)
        review_date = start + pd.Timedelta(days=rng.randint(0, span_days))
        rows.append(
            {
                "review_id": i,
                "review_date": review_date.date().isoformat(),
                "product_category": category,
                "rating": rating,
                "review_text": text,
            }
        )

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "data" / "reviews.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} synthetic reviews -> {out}")
    print(df["rating"].value_counts().sort_index().to_string())
    print(df["product_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
