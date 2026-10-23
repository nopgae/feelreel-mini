# FeelReel-Mini: Multimodal Emotion Understanding Pipeline

A small-scale personal recreation of the data pipeline behind **FeelReel**, a multimodal emotion dataset project I worked on as a Research Assistant at Penn State (Jun 2024 – Mar 2025).

> **This is not the lab's code or data.** This is a separate personal project built with public tools (PRAW, a public GoEmotions RoBERTa checkpoint, LLaVA via Ollama) to show how the pipeline works at a small scale. The original dataset and code are not public.

## About the Original Project

FeelReel is a benchmark of 400K+ Reddit images with 5M+ user comments, annotated with emotion labels and visually grounded explanations without human annotators. An earlier version of the paper was submitted to CVPR 2024 (before I joined); it has not been published. I joined during the follow-up phase.

## My Role

- **Data collection & organization** — sourced Reddit image posts and comments through the project's collection pipeline and organized them into structured datasets
- **Human evaluation (MTurk)** — built the MTurk survey (HIT) page and collected human evaluation responses on the dataset's emotion annotations through it

I did not train or fine-tune the project's models, and the experiments in the original paper predate my involvement.

---

## Pipeline Overview

```
Reddit (image posts + comments)
        │
        ▼
[1] Scraper           — PRAW, requires free Reddit API credentials
        │
        ▼
[2] GoEmotions        — 28 fine-grained emotion labels per comment
        │
        ▼
[3] LLaVA (optional)  — "what in this image triggered [emotion]?"
        │
        ▼
[4] Analysis          — emotion distribution charts + sample outputs
```

This mirrors the attribute extraction pipeline from the original project:
- **GoEmotions** for pseudo emotion labeling
- **LLaVA** for emotion stimulus extraction via visual question answering
- Distribution charts in the style of the original project's analysis

---

## Results

Emotion distribution across image-heavy subreddits (demo run, 8 subreddits):

![Emotion Distribution](sample_outputs/emotion_distribution.png)

Emotion patterns vary meaningfully by community:

![Subreddit Heatmap](sample_outputs/subreddit_emotion_heatmap.png)

Key finding: visual subreddits (r/earthporn, r/woodworking) skew toward **admiration** and **surprise**, while pet communities (r/rarepuppers) show more **joy** — which is the kind of content-dependent pattern FeelReel was designed to capture. This is a small demo run, so treat it as illustrative rather than a finding.

---

## Setup

```bash
git clone https://github.com/nopgae/feelreel-mini.git
cd feelreel-mini
pip install -r requirements.txt
```

**Reddit API credentials (required for scraping):**
```bash
# 1. Go to https://www.reddit.com/prefs/apps → "create another app" → type: script
# 2. Set environment variables:
export REDDIT_CLIENT_ID=your_client_id
export REDDIT_CLIENT_SECRET=your_client_secret
```

**For LLaVA stimulus extraction (optional):**
```bash
brew install ollama
ollama pull llava
ollama serve   # run in a separate terminal
```

---

## Usage

```bash
# Full pipeline: scrape → label → visualize
python run_pipeline.py

# Skip re-scraping (reuse saved data)
python run_pipeline.py --skip-scrape

# Also run LLaVA emotion stimulus extraction
python run_pipeline.py --with-stimulus
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Reddit scraping | PRAW (requires free Reddit API credentials) |
| Emotion labeling | `SamLowe/roberta-base-go_emotions` (RoBERTa, 28 labels) |
| Emotion stimulus | LLaVA 7B via Ollama (Metal GPU on Apple Silicon) |
| Visualization | matplotlib, seaborn |

---

## Project Structure

```
feelreel-mini/
├── scraper/              # Reddit image post + comment collection
├── annotation/           # GoEmotions pseudo-labeling
├── stimulus/             # LLaVA visual emotion stimulus extraction
├── analysis/             # Charts and sample output generation
├── data/                 # Collected data (gitignored)
└── outputs/              # Generated charts and tables
```

---

## Background

The original FeelReel dataset addresses a gap in affective computing: existing emotion datasets either focus narrowly on facial expressions or rely on expensive manual annotation. FeelReel leverages real user reactions to real-world images at scale, capturing a more authentic distribution of human emotional responses.

This mini version demonstrates the same core insight: that social media comment data, combined with modern vision-language models, can produce rich, interpretable emotion annotations without human labelers.
