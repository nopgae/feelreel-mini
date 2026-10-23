"""
GoEmotions Dataset Loader
Loads the Google GoEmotions dataset (211K real Reddit comments) from HuggingFace
as a drop-in replacement for the Reddit scraper when API access is unavailable.

This also enables a bonus analysis: comparing our model's predictions
against the human-labeled ground truth in the dataset.
"""

import pandas as pd
from pathlib import Path
from datasets import load_dataset

EMOTION_COLS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# Visual/image-adjacent subreddits present in the GoEmotions dataset
VISUAL_SUBREDDITS = {
    "pics", "funny", "aww", "gaming", "mildlyinteresting", "food",
    "todayilearned", "worldnews", "science", "movies", "television",
    "sports", "nba", "soccer", "nfl", "baseball",
}


def load(
    n_comments: int = 5000,
    output_comments: Path = Path("data/comments.csv"),
    output_posts: Path = Path("data/posts.csv"),
    filter_visual: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("  Downloading GoEmotions dataset from HuggingFace...")
    ds = load_dataset("go_emotions", "raw", split="train")
    df = ds.to_pandas()

    # Deduplicate: each comment has multiple rater rows — keep first annotation
    df = df.drop_duplicates(subset="id")

    # Human ground-truth label: the emotion column with value 1 (take first if multiple)
    def _human_label(row: pd.Series) -> str:
        active = [e for e in EMOTION_COLS if row.get(e, 0) == 1]
        return active[0] if active else "neutral"

    df["human_label"] = df.apply(_human_label, axis=1)

    if filter_visual:
        mask = df["subreddit"].str.lower().isin(VISUAL_SUBREDDITS)
        df_filtered = df[mask]
        if len(df_filtered) >= n_comments // 2:
            df = df_filtered
            print(f"  Filtered to visual subreddits: {len(df):,} comments")

    df = df.sample(n=min(n_comments, len(df)), random_state=42)
    print(f"  Using {len(df):,} comments from {df['subreddit'].nunique()} subreddits")

    df_comments = pd.DataFrame({
        "post_id":    df["link_id"].str.replace("t3_", "", regex=False),
        "comment_id": df["id"],
        "text":       df["text"],
        "score":      0,
        "subreddit":  df["subreddit"],
        "human_label": df["human_label"],
    })

    # Synthesize posts table from unique link_ids
    df_posts = (
        df_comments[["post_id", "subreddit"]]
        .drop_duplicates(subset="post_id")
        .assign(
            title="",
            image_url="",
            image_path="",
            score=0,
            num_comments=0,
            created_utc=0,
        )
    )

    df_comments.to_csv(output_comments, index=False)
    df_posts.to_csv(output_posts, index=False)
    print(f"  Saved: {len(df_posts)} posts, {len(df_comments)} comments")

    return df_posts, df_comments
