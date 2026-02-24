"""
Data loading, cleaning, and feature engineering for Spotify user behavior
analysis. Transforms the raw survey data into analysis-ready features
for regression modeling.
"""

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv


# Ordinal mappings for engineering continuous proxies

AGE_MIDPOINTS = {
    "6-12": 9,       # [6, 12]
    "12-20": 16,     # [13, 20]
    "20-35": 28,     # [21, 35]
    "35-60": 48,     # [36, 60]
    "60+": 70,       # [61, 80] assumed upper bound
}

USAGE_PERIOD_MONTHS = {
    "Less than 6 months": 3,
    "6 months to 1 year": 9,
    "1 year to 2 years": 18,
    "More than 2 years": 30,
}

POD_FREQUENCY_SCORE = {
    "Never": 0,
    "Rarely": 1,
    "Once a week": 2,
    "Several times a week": 3,
    "Daily": 4,
}

TIME_SLOT_HOURS = {
    "Morning": 2.0,
    "Afternoon": 1.5,
    "Night": 2.5,
}


def load_raw_data(path):
    """Read the raw Excel file from disk."""
    return pd.read_excel(path)


def clean_data(df):
    """Standardize column names, fix typos, and handle missing values."""
    df = df.copy()

    # Standardize column names to snake_case
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Standardize genre values
    df["fav_music_genre"] = (
        df["fav_music_genre"]
        .str.strip()
        .str.title()
        .replace({
            "Classical": "Classical",
            "Classical & Melody, Dance": "Classical",
            "Old Songs": "Pop",
            "Trending Songs Random": "Pop",
            "Kpop": "Pop",
            "All": "Melody",
        })
    )

    # Fill missing podcast columns with "Unknown"
    podcast_cols = [
        "fav_pod_genre",
        "preffered_pod_format",
        "pod_host_preference",
        "preffered_pod_duration",
    ]
    for col in podcast_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Fill missing premium plan with "None"
    if "preffered_premium_plan" in df.columns:
        df["preffered_premium_plan"] = (
            df["preffered_premium_plan"].fillna("None")
        )

    return df


def _count_listening_contexts(series):
    """Count how many listening contexts a user selected (comma-separated)."""
    return series.str.split(",").apply(len)


def _compute_genre_diversity(series):
    """
    Assign a diversity score based on genre specificity.
    Users who picked broad/generic genres get a higher score.
    Users who picked very specific niche genres get a lower score.
    """
    broad_genres = {"Melody", "Pop", "All"}
    return series.apply(lambda g: 1 if g in broad_genres else 0)


def engineer_features(df):
    """
    Create continuous proxy variables from categorical survey data.

    Engineered variables:
        age_numeric: Midpoint of the age bracket
        listening_time: Estimated daily listening minutes based on
            number of listening contexts and time slot
        skip_rate: Inverse of recommendation rating (1-5 scale inverted)
        diversity_score: Number of unique listening contexts as a
            proxy for behavioral variety
        streams: Composite usage intensity score combining listening
            contexts, usage period, and pod frequency
        bot_like: Binary label where accounts with high usage intensity,
            low diversity, and extreme listening patterns are flagged
    """
    df = df.copy()

    # Age as numeric midpoint
    df["age_numeric"] = df["age"].map(AGE_MIDPOINTS)

    # Count of listening contexts (proxy for usage breadth)
    df["n_listening_contexts"] = _count_listening_contexts(
        df["music_lis_frequency"]
    )

    # Estimated daily listening time (minutes)
    # More contexts and later time slots suggest more daily listening
    base_minutes = df["music_time_slot"].map(TIME_SLOT_HOURS) * 60
    df["listening_time"] = (
        base_minutes * (1 + 0.15 * (df["n_listening_contexts"] - 1))
    )
    # Add noise for variance (seeded for reproducibility)
    rng = np.random.default_rng(seed=42)
    df["listening_time"] += rng.normal(0, 10, size=len(df))
    df["listening_time"] = df["listening_time"].clip(lower=10).round(1)

    # Skip rate: inverse of recommendation satisfaction (1-5 scale)
    # Higher rating means user finds good content (lower skip rate)
    df["skip_rate"] = (6 - df["music_recc_rating"]) / 5

    # Diversity score: number of listening contexts normalized
    max_contexts = df["n_listening_contexts"].max()
    df["diversity_score"] = (
        df["n_listening_contexts"] / max_contexts
    ).round(3)

    # Usage period in months
    df["usage_months"] = df["spotify_usage_period"].map(USAGE_PERIOD_MONTHS)

    # Podcast frequency as numeric
    df["pod_frequency"] = df["pod_lis_frequency"].map(POD_FREQUENCY_SCORE)

    # Streams: composite usage intensity score
    # Combines how long on platform, how many contexts, and overall engagement
    df["streams"] = (
        df["usage_months"] * 50
        + df["n_listening_contexts"] * 200
        + df["listening_time"] * 2
        + df["pod_frequency"] * 100
    ).round(0)
    # Add noise
    df["streams"] += rng.normal(0, 150, size=len(df)).round(0)
    df["streams"] = df["streams"].clip(lower=50).astype(int)

    # Bot-like label: flag accounts with extreme usage patterns
    # High streams + low diversity + low recommendation engagement
    high_streams = df["streams"] > df["streams"].quantile(0.75)
    low_diversity = df["diversity_score"] <= df["diversity_score"].quantile(0.25)
    low_recc = df["music_recc_rating"] <= 2
    df["bot_like"] = (
        (high_streams & low_diversity)
        | (high_streams & low_recc)
        | (low_diversity & low_recc)
    ).astype(int)

    return df


def create_dummies(df, columns):
    """
    Create dummy variables for specified categorical columns.
    Drops the first category to avoid multicollinearity.
    """
    return pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)


def load_and_clean_data(path):
    """
    Full pipeline: load raw data, clean, engineer features,
    and return the analysis-ready dataframe.
    """
    df = load_raw_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    return df
