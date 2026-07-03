"""
Reddit Image Post Scraper
Collects image posts and top comments from visual subreddits.

Requires Reddit API credentials (free):
  1. Go to https://www.reddit.com/prefs/apps → "create another app" → type: script
  2. Set environment variables:
       export REDDIT_CLIENT_ID=your_client_id
       export REDDIT_CLIENT_SECRET=your_client_secret

Usage:
  from scraper.reddit_scraper import scrape
  df_posts, df_comments = scrape()
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
IMAGE_DOMAINS = {"i.redd.it", "i.imgur.com", "preview.redd.it"}

SUBREDDITS = [
    "pics", "mildlyinteresting", "itookapicture",
    "food", "foodporn", "baking",
    "earthporn", "gardening",
    "rarepuppers", "aww",
    "woodworking", "houseplants",
]

HEADERS = {"User-Agent": "feelreel-mini/1.0 by research-portfolio"}


def _is_image_url(url: str) -> bool:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain_ok = any(parsed.netloc == d or parsed.netloc.endswith("." + d) for d in IMAGE_DOMAINS)
    ext_ok = Path(parsed.path.split("?")[0]).suffix.lower() in IMAGE_EXTENSIONS
    return domain_ok or ext_ok


def _download_image(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20, stream=True)
        if r.status_code != 200:
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def _fetch_with_praw(
    subreddits: list[str],
    posts_per_sub: int,
    comments_per_post: int,
    images_dir: Path,
) -> tuple[list[dict], list[dict]]:
    import praw

    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise EnvironmentError(
            "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set.\n"
            "Get credentials at: https://www.reddit.com/prefs/apps"
        )

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent="feelreel-mini/1.0 by research-portfolio",
    )

    all_posts: list[dict] = []
    all_comments: list[dict] = []

    for sub in subreddits:
        count = 0
        for post in reddit.subreddit(sub).top(time_filter="year", limit=posts_per_sub * 3):
            if post.stickied or post.is_video:
                continue
            if not _is_image_url(post.url):
                continue

            ext = Path(post.url.split("?")[0]).suffix.lower() or ".jpg"
            if ext not in IMAGE_EXTENSIONS:
                ext = ".jpg"
            img_path = images_dir / f"{post.id}{ext}"
            if not img_path.exists():
                _download_image(post.url, img_path)

            all_posts.append({
                "post_id":      post.id,
                "subreddit":    sub,
                "title":        post.title.strip(),
                "image_url":    post.url,
                "image_path":   str(img_path) if img_path.exists() else "",
                "score":        post.score,
                "num_comments": post.num_comments,
                "created_utc":  int(post.created_utc),
            })

            post.comments.replace_more(limit=0)
            for comment in list(post.comments)[:comments_per_post]:
                body = comment.body.strip()
                if body in ("[deleted]", "[removed]") or len(body.split()) < 3:
                    continue
                all_comments.append({
                    "post_id":    post.id,
                    "comment_id": comment.id,
                    "text":       body,
                    "score":      comment.score,
                })

            count += 1
            if count >= posts_per_sub:
                break
            time.sleep(0.5)

        print(f"  r/{sub}: {count} image posts")

    return all_posts, all_comments


def scrape(
    subreddits: list[str] = SUBREDDITS,
    posts_per_sub: int = 30,
    comments_per_post: int = 8,
    images_dir: Path = Path("data/images"),
    output_posts: Path = Path("data/posts.csv"),
    output_comments: Path = Path("data/comments.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        import praw  # noqa: F401
    except ImportError:
        raise ImportError("Install praw: pip install praw")

    all_posts, all_comments = _fetch_with_praw(
        subreddits, posts_per_sub, comments_per_post, images_dir
    )

    df_posts = pd.DataFrame(all_posts).drop_duplicates(subset="post_id")
    df_comments = (
        pd.DataFrame(all_comments).drop_duplicates(subset="comment_id")
        if all_comments else pd.DataFrame()
    )

    df_posts.to_csv(output_posts, index=False)
    df_comments.to_csv(output_comments, index=False)

    print(f"\n  Saved: {len(df_posts)} posts, {len(df_comments)} comments")
    return df_posts, df_comments
