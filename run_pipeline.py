"""
FeelReel-Mini Pipeline

Usage:
    python run_pipeline.py                   # full pipeline (requires Reddit credentials)
    python run_pipeline.py --skip-scrape     # reuse existing data/posts.csv
    python run_pipeline.py --demo            # run on built-in sample data (no credentials)
    python run_pipeline.py --with-stimulus   # also run LLaVA extraction (requires Ollama)

Reddit credentials (for scraping):
    export REDDIT_CLIENT_ID=your_id
    export REDDIT_CLIENT_SECRET=your_secret
    Get credentials at: https://www.reddit.com/prefs/apps
"""

import argparse
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
POSTS_CSV = DATA_DIR / "posts.csv"
COMMENTS_CSV = DATA_DIR / "comments.csv"
LABELED_CSV = DATA_DIR / "comments_labeled.csv"
STIMULUS_CSV = DATA_DIR / "comments_with_stimulus.csv"

SAMPLE_COMMENTS = [
    ("abc01", "pics",             "Golden hour at the Grand Canyon",           "This is absolutely breathtaking. I need to visit someday."),
    ("abc01", "pics",             "Golden hour at the Grand Canyon",           "The colors are unreal, looks like a painting!"),
    ("abc01", "pics",             "Golden hour at the Grand Canyon",           "Nature truly is the best artist."),
    ("abc02", "food",             "Homemade ramen from scratch",               "This looks incredible! Did you make the broth yourself?"),
    ("abc02", "food",             "Homemade ramen from scratch",               "I'm genuinely jealous, that looks so comforting."),
    ("abc02", "food",             "Homemade ramen from scratch",               "The egg is cooked perfectly, I can never get mine right."),
    ("abc03", "mildlyinteresting","This mushroom grew in a perfect spiral",    "Nature never ceases to amaze me."),
    ("abc03", "mildlyinteresting","This mushroom grew in a perfect spiral",    "That's actually terrifying to me for some reason."),
    ("abc03", "mildlyinteresting","This mushroom grew in a perfect spiral",    "I wonder how long it took to grow like that."),
    ("abc04", "rarepuppers",      "My dog's first snow experience",            "The pure joy on that face!!"),
    ("abc04", "rarepuppers",      "My dog's first snow experience",            "I'm crying, this is too wholesome."),
    ("abc04", "rarepuppers",      "My dog's first snow experience",            "This made my whole day honestly."),
    ("abc05", "earthporn",        "Northern lights over Norway",               "This literally doesn't look real. Stunning."),
    ("abc05", "earthporn",        "Northern lights over Norway",               "It's on my bucket list. One day..."),
    ("abc05", "earthporn",        "Northern lights over Norway",               "I had no idea the sky could look like that."),
    ("abc06", "woodworking",      "Dining table I built over 3 months",        "The craftsmanship here is incredible. Respect."),
    ("abc06", "woodworking",      "Dining table I built over 3 months",        "What kind of wood is that? The grain is beautiful."),
    ("abc06", "woodworking",      "Dining table I built over 3 months",        "This must have taken so much patience. Amazing work."),
    ("abc07", "baking",           "First attempt at croissants",               "For a first attempt?! These look professional."),
    ("abc07", "baking",           "First attempt at croissants",               "The layers are perfect. I tried once and failed miserably."),
    ("abc07", "baking",           "First attempt at croissants",               "Can you share the recipe? Please!"),
    ("abc08", "itookapicture",    "Old man feeding pigeons in the rain",       "This is a genuinely moving photograph."),
    ("abc08", "itookapicture",    "Old man feeding pigeons in the rain",       "The composition here is really special. Great shot."),
    ("abc08", "itookapicture",    "Old man feeding pigeons in the rain",       "This makes me feel lonely and warm at the same time."),
]

SAMPLE_POSTS = [
    ("abc01", "pics",             "Golden hour at the Grand Canyon",           "https://i.redd.it/sample1.jpg", 45000, 312),
    ("abc02", "food",             "Homemade ramen from scratch",               "https://i.redd.it/sample2.jpg", 32000, 218),
    ("abc03", "mildlyinteresting","This mushroom grew in a perfect spiral",    "https://i.redd.it/sample3.jpg", 18000, 145),
    ("abc04", "rarepuppers",      "My dog's first snow experience",            "https://i.redd.it/sample4.jpg", 67000, 430),
    ("abc05", "earthporn",        "Northern lights over Norway",               "https://i.redd.it/sample5.jpg", 89000, 521),
    ("abc06", "woodworking",      "Dining table I built over 3 months",        "https://i.redd.it/sample6.jpg", 24000, 187),
    ("abc07", "baking",           "First attempt at croissants",               "https://i.redd.it/sample7.jpg", 15000, 102),
    ("abc08", "itookapicture",    "Old man feeding pigeons in the rain",       "https://i.redd.it/sample8.jpg", 41000, 298),
]


def _build_sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_posts = pd.DataFrame(SAMPLE_POSTS, columns=[
        "post_id", "subreddit", "title", "image_url", "score", "num_comments",
    ])
    df_posts["image_path"] = ""
    df_posts["created_utc"] = 1700000000

    rows = []
    for i, (pid, sub, title, text) in enumerate(SAMPLE_COMMENTS):
        rows.append({"post_id": pid, "comment_id": f"c{i:03d}", "text": text, "score": 100})
    df_comments = pd.DataFrame(rows)

    DATA_DIR.mkdir(exist_ok=True)
    df_posts.to_csv(POSTS_CSV, index=False)
    df_comments.to_csv(COMMENTS_CSV, index=False)
    print(f"  Demo data: {len(df_posts)} posts, {len(df_comments)} comments")
    return df_posts, df_comments


def run(
    skip_scrape: bool = False,
    demo: bool = False,
    real_data: bool = False,
    with_stimulus: bool = False,
) -> None:
    # ── Step 1: Data ──────────────────────────────────────────────────────────
    if demo:
        print("\n[1/4] Using built-in sample data (demo mode)...")
        df_posts, df_comments = _build_sample_data()
    elif real_data:
        print("\n[1/4] Loading GoEmotions dataset (211K real Reddit comments)...")
        from scraper.goemotions_loader import load
        df_posts, df_comments = load(
            n_comments=5000,
            output_comments=COMMENTS_CSV,
            output_posts=POSTS_CSV,
        )
    elif skip_scrape:
        print("\n[1/4] Loading existing scraped data...")
        df_posts = pd.read_csv(POSTS_CSV)
        df_comments = pd.read_csv(COMMENTS_CSV)
        print(f"  {len(df_posts)} posts, {len(df_comments)} comments loaded")
    else:
        print("\n[1/4] Scraping Reddit image posts and comments...")
        from scraper.reddit_scraper import scrape
        df_posts, df_comments = scrape(
            output_posts=POSTS_CSV,
            output_comments=COMMENTS_CSV,
        )

    if df_comments.empty:
        print("  No comments found. Exiting.")
        return

    # ── Step 2: Emotion Labeling ──────────────────────────────────────────────
    print("\n[2/4] Running GoEmotions on comments...")
    from annotation.emotion_labeler import load_model, label_comments
    classifier = load_model()
    df_labeled = label_comments(df_comments, classifier, output_path=LABELED_CSV)

    # ── Step 3: Stimulus Extraction (optional) ────────────────────────────────
    if with_stimulus:
        print("\n[3/4] Extracting emotion stimuli via LLaVA...")
        from stimulus.vlm_extractor import extract_stimuli
        df_labeled = extract_stimuli(df_labeled, df_posts, output_path=STIMULUS_CSV)
    else:
        print("\n[3/4] Skipping stimulus extraction (use --with-stimulus + Ollama to enable)")

    # ── Step 4: Visualize ─────────────────────────────────────────────────────
    print("\n[4/4] Generating visualizations...")
    from analysis.visualize import (
        plot_emotion_distribution,
        plot_subreddit_emotion_heatmap,
        plot_model_vs_human,
        save_sample_outputs,
        print_stats,
    )
    plot_emotion_distribution(df_labeled)
    plot_subreddit_emotion_heatmap(df_labeled, df_posts)
    if "human_label" in df_labeled.columns:
        plot_model_vs_human(df_labeled)
    save_sample_outputs(df_labeled, df_posts)
    print_stats(df_labeled, df_posts)

    print("Done. Outputs saved to outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FeelReel-Mini pipeline")
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip scraping, load existing data/posts.csv + data/comments.csv",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run on built-in sample data — no Reddit credentials needed",
    )
    parser.add_argument(
        "--with-stimulus", action="store_true",
        help="Run LLaVA emotion stimulus extraction (requires Ollama + llava model)",
    )
    parser.add_argument(
        "--real-data", action="store_true",
        help="Load 5K real Reddit comments from GoEmotions dataset (HuggingFace, no credentials)",
    )
    args = parser.parse_args()
    run(
        skip_scrape=args.skip_scrape,
        demo=args.demo,
        real_data=args.real_data,
        with_stimulus=args.with_stimulus,
    )
