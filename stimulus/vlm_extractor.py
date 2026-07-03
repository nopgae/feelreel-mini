"""
Emotion Stimulus Extractor
Uses LLaVA (via Ollama) to answer: "what in this image triggered [emotion]?"
This mirrors the LLaVA query step from the FeelReel pipeline.

Requires Ollama running locally with the llava model pulled:
    brew install ollama
    ollama pull llava
    ollama serve
"""

import base64
import json
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llava"

PROMPT_TEMPLATE = (
    "This is an image from a Reddit post titled: \"{title}\". "
    "A user commented: \"{comment}\". "
    "What is triggering the {emotion} emotion in this image? "
    "Give a simple and concise answer in one sentence."
)


def _is_ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _encode_image(image_path: str) -> str | None:
    path = Path(image_path)
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _query_llava(image_b64: str, prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[error: {e}]"


def extract_stimuli(
    df_labeled: pd.DataFrame,
    df_posts: pd.DataFrame,
    max_comments: int | None = None,
    output_path: Path = Path("data/comments_with_stimulus.csv"),
) -> pd.DataFrame:
    if not _is_ollama_available():
        print(
            "  Ollama not running. Skipping stimulus extraction.\n"
            "  To enable: brew install ollama && ollama pull llava && ollama serve"
        )
        df_labeled["stimulus"] = ""
        return df_labeled

    post_map = df_posts.set_index("post_id")[["title", "image_path"]].to_dict("index")

    df = df_labeled.copy()
    if max_comments:
        df = df.head(max_comments)

    stimuli: list[str] = []
    print(f"  Extracting emotion stimuli for {len(df)} comments via LLaVA...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        post = post_map.get(row["post_id"], {})
        image_path = post.get("image_path", "")
        title = post.get("title", "")

        image_b64 = _encode_image(image_path)
        if not image_b64:
            stimuli.append("")
            continue

        prompt = PROMPT_TEMPLATE.format(
            title=title,
            comment=row["text"][:200],
            emotion=row["emotion"],
        )
        stimulus = _query_llava(image_b64, prompt)
        stimuli.append(stimulus)

    df["stimulus"] = stimuli
    df.to_csv(output_path, index=False)
    print(f"  Saved stimuli → {output_path}")
    return df
