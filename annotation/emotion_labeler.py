"""
GoEmotions Emotion Labeler
Assigns one of 28 fine-grained emotion labels to each Reddit comment,
mirroring the pseudo-labeling step from the FeelReel pipeline.

Model: SamLowe/roberta-base-go_emotions (28 categories, runs on CPU/MPS)
"""

import torch
import pandas as pd
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm

MODEL_ID = "SamLowe/roberta-base-go_emotions"

EMOTIONS_28 = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model() -> pipeline:
    device = _get_device()
    print(f"  Loading GoEmotions model on {device}...")
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        top_k=1,
        device=device,
        truncation=True,
        max_length=512,
    )
    return classifier


def label_comments(
    df_comments: pd.DataFrame,
    classifier: pipeline,
    batch_size: int = 32,
    output_path: Path = Path("data/comments_labeled.csv"),
) -> pd.DataFrame:
    texts = df_comments["text"].tolist()
    labels: list[str] = []
    scores: list[float] = []

    print(f"  Labeling {len(texts)} comments...")
    for i in tqdm(range(0, len(texts), batch_size), unit="batch"):
        batch = texts[i : i + batch_size]
        results = classifier(batch)
        for result in results:
            top = result[0]
            labels.append(top["label"])
            scores.append(round(top["score"], 4))

    df_out = df_comments.copy()
    df_out["emotion"] = labels
    df_out["emotion_score"] = scores
    df_out.to_csv(output_path, index=False)
    print(f"  Saved labeled comments → {output_path}")
    return df_out


def emotion_summary(df_labeled: pd.DataFrame) -> pd.Series:
    return df_labeled["emotion"].value_counts(normalize=True).round(4)
