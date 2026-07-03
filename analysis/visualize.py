"""
Visualization and Analysis
Generates emotion distribution charts and sample output tables,
mirroring the analysis in the FeelReel paper (Figures 3, 4, 5).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

EMOTION_ORDER = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

POSITIVE = {
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief",
}
NEGATIVE = {
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
}


def _emotion_color(emotion: str) -> str:
    if emotion in POSITIVE:
        return "#4CAF50"
    if emotion in NEGATIVE:
        return "#F44336"
    return "#9E9E9E"


def plot_emotion_distribution(
    df_labeled: pd.DataFrame,
    output_path: Path = Path("outputs/emotion_distribution.png"),
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = df_labeled["emotion"].value_counts()
    present = [e for e in EMOTION_ORDER if e in counts.index]
    values = [counts[e] / counts.sum() * 100 for e in present]
    colors = [_emotion_color(e) for e in present]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(present)), values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Emotion Distribution in FeelReel-Mini\n(GoEmotions, 28 categories)", fontsize=13)
    ax.set_ylim(0, max(values) * 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="Positive"),
        mpatches.Patch(color="#F44336", label="Negative"),
        mpatches.Patch(color="#9E9E9E", label="Neutral/Ambiguous"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {output_path}")


def plot_subreddit_emotion_heatmap(
    df_labeled: pd.DataFrame,
    df_posts: pd.DataFrame,
    output_path: Path = Path("outputs/subreddit_emotion_heatmap.png"),
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = df_labeled.merge(df_posts[["post_id", "subreddit"]], on="post_id", how="left")
    pivot = (
        df.groupby(["subreddit", "emotion"])
        .size()
        .unstack(fill_value=0)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    top_emotions = df["emotion"].value_counts().head(12).index.tolist()
    pivot_pct = pivot_pct[[e for e in top_emotions if e in pivot_pct.columns]]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        pivot_pct,
        ax=ax,
        cmap="YlOrRd",
        fmt=".1f",
        annot=True,
        annot_kws={"size": 7},
        linewidths=0.4,
        cbar_kws={"label": "% of comments"},
    )
    ax.set_title("Emotion Distribution by Subreddit (%)", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {output_path}")


def save_sample_outputs(
    df_labeled: pd.DataFrame,
    df_posts: pd.DataFrame,
    n_per_emotion: int = 2,
    output_path: Path = Path("outputs/sample_outputs.csv"),
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = df_labeled.merge(df_posts[["post_id", "subreddit", "title"]], on="post_id", how="left")
    top_emotions = df["emotion"].value_counts().head(10).index.tolist()

    samples = (
        df[df["emotion"].isin(top_emotions)]
        .groupby("emotion", group_keys=False)
        .apply(lambda x: x.nlargest(n_per_emotion, "emotion_score"))
    )
    cols = ["subreddit", "title", "text", "emotion", "emotion_score"]
    if "stimulus" in samples.columns:
        cols.append("stimulus")

    samples[cols].to_csv(output_path, index=False)
    print(f"  Saved → {output_path}")


def print_stats(df_labeled: pd.DataFrame, df_posts: pd.DataFrame) -> None:
    print(f"\n{'─'*50}")
    print(f"  Posts collected:     {len(df_posts)}")
    print(f"  Comments collected:  {len(df_labeled)}")
    print(f"  Subreddits:          {df_posts['subreddit'].nunique()}")
    print(f"  Unique emotions:     {df_labeled['emotion'].nunique()}")
    print(f"\n  Top 5 emotions:")
    dist = df_labeled["emotion"].value_counts(normalize=True) * 100
    for emotion, pct in dist.head(5).items():
        print(f"    {emotion:<20} {pct:.1f}%")
    print(f"{'─'*50}\n")
