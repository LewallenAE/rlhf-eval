# RLHF Evaluation Harness

[![CI](https://github.com/LewallenAE/rlhf-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/LewallenAE/rlhf-eval/actions/workflows/ci.yml)

An end-to-end system for detecting problematic preference pairs in RLHF training data, training reward models on filtered vs. unfiltered datasets, and measuring the impact of data quality on model performance.

Built against Anthropic's [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset (160,800 preference pairs).

## Motivation

Reward models are only as good as the preference data they're trained on. Noisy labels, near-duplicate responses, degenerate text, and labeling bias all degrade reward signal quality. This project identifies and removes those failure modes, then empirically validates the impact by training competing reward models on clean vs. unfiltered data.

## Results

**Data Quality Pipeline** — 160,800 examples scored by 7 detectors:

| Detector | Flagged | Rate | What It Catches |
|---|---|---|---|
| Semantic Similarity | 804 | 0.5% | Near-identical chosen/rejected pairs |
| Readability Mismatch | 8,041 | 5.0% | Rejected response has better readability |
| Repetition | 1,610 | 1.0% | Degenerate repeated text |
| Length Ratio | 8,067 | 5.0% | Suspiciously short responses to complex prompts |
| Refusal Bias | 291 | 0.2% | Chosen refuses while rejected is helpful |
| Unsafe Prompt | 1,680 | 1.0% | Prompts where neither response may be valid |
| LLM Judge (GPT-4o-mini) | TBD | TBD | Semantic pathologies rule-based detectors miss |
| **Total unique** | **12,693+** | **7.9%+** | |

**148,107 clean examples** retained (92.1% of dataset).

**Reward Model Comparison** —> DistilBERT (67M params), Bradley-Terry loss, 1 epoch on T4 GPU:

| Metric | Clean (filtered) | Unfiltered |
|---|---|---|
| Training examples | 148,107 | 160,800 |
| Test accuracy | 62.18% | 62.33% |
| Avg reward gap (chosen − rejected) | 0.2282 | 0.3014 |

![Reward Model Comparison](src/results/Preference_Pair_Test_Result_01.webp)
![Reward Model Comparison](src/results/Preference_Pair_Test_Result_02.webp)

### Interpreting the Results

Binary accuracy is nearly identical across both models (62.18% vs. 62.33%) — and that's the point. The unfiltered model's larger reward gap (0.3014 vs. 0.2282) looks like stronger performance on the surface, but it reflects inflated confidence learned from degenerate pairs: near-duplicates where the label is arbitrary, readability-mismatched examples where the "rejected" response is objectively better, and refusal bias cases where the chosen response actively refuses a reasonable request.

The clean model produces a tighter, more calibrated reward gap. In production RLHF pipelines, overconfident reward signals from noisy data are a known driver of reward hacking: the model learns to exploit the noise rather than the signal. Filtering 7.9% of pathological examples doesn't hurt accuracy; it removes the false confidence that makes reward models brittle.

**Binary accuracy is the wrong metric here. Reward gap calibration is the right one.**

## Real Examples ##

Chosen: 36 words | Rejected: 1 word | Ratio: 0.028
The length ratio detector flagged this pair, but the real pathology isn't length, it's refusal behavior. The chosen response engages with a prompt that warrants a clear safety boundary, hedging instead of refusing. The rejected response ("Words?") is unhelpful, but the chosen response is actively worse: it teaches the reward model that circumventing a refusal is preferable to setting one.
This is a true positive for the wrong stated reason and exactly the class of label noise an LLM judge catches that rule-based detectors miss.

<img width="1834" height="791" alt="image" src="https://github.com/user-attachments/assets/6d69dd67-9399-400c-8631-5331945a3241" />


## Architecture

```
src/rlhf_eval/
├── api/
│   └── routes.py              # FastAPI service (POST /ingest, POST /score, GET /experiments)
├── config/
│   └── settings.py            # Pydantic configuration (RLHF_ env prefix)
├── database/
│   ├── models.py              # SQLAlchemy ORM models (5 tables)
│   ├── connection.py          # Engine, session, context manager
│   └── operations.py          # CRUD operations
├── detectors/
│   ├── base.py                # Abstract base class with threshold logic
│   ├── semantic_similarity.py # sentence-transformers cosine similarity
│   ├── readability.py         # textstat Flesch-Kincaid comparison
│   ├── repetition.py          # Unique word ratio + n-gram detection
│   ├── length_ratio.py        # Response-to-prompt length ratio
│   ├── refusal_bias.py        # Refusal pattern matching
│   ├── unsafe_prompt.py       # Toxicity keyword classification
│   └── llm_judge.py           # GPT-4o-mini semantic label evaluation
├── pipeline/
│   ├── data_loader.py         # HuggingFace dataset ingestion
│   └── quality_pipeline.py    # Orchestrates detector runs
├── reward/
│   ├── model.py               # RewardModel (DistilBERT + linear head)
│   ├── dataset.py             # PreferencePairDataset + data loaders
│   ├── train.py               # Bradley-Terry pairwise loss training
│   └── evaluate.py            # Accuracy + reward gap evaluation
└── utils/
    ├── parsing.py             # HH-RLHF conversation parsing
    └── stats.py               # Statistical utilities

dashboard/
└── app.py                     # Streamlit dashboard (flagged explorer + RM comparison)

scripts/
└── export_flagged_indices.py  # Export flagged indices for Colab

notebooks/
└── reward_model_experiment.ipynb  # Self-contained Colab notebook

tests/                         # pytest suite across 5 test modules
docker-compose.yml             # postgres + api + dashboard
Dockerfile
.github/workflows/ci.yml       # CI: pytest + mypy + ruff
```

## Quick Start

### Docker Compose (recommended)

```bash
git clone https://github.com/LewallenAE/rlhf-eval.git
cd rlhf-eval
OPENAI_API_KEY=your-key docker-compose up
```

This starts PostgreSQL, the FastAPI service (`http://localhost:8000`), and the Streamlit dashboard (`http://localhost:8501`).

- **API docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501

### Manual Setup

#### Prerequisites

- Python 3.11+
- PostgreSQL 16

#### Installation

```bash
git clone https://github.com/LewallenAE/rlhf-eval.git
cd rlhf-eval
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

#### Environment

Create a `.env` file in the project root:

```
RLHF_DATABASE_URL=postgresql://user:pass@localhost:5432/rlhf_dev
OPENAI_API_KEY=your-key-here   # required only for LLM judge detector
```

#### Run Tests

```bash
pytest tests/ -v
```

## Usage

### 1. Data Ingestion & Quality Scoring

The pipeline loads HH-RLHF from HuggingFace, parses conversations, ingests into PostgreSQL, and runs all 6 detectors:

```python
from rlhf_eval.database.connection import get_engine, SessionContext
from rlhf_eval.database.models import Base
from rlhf_eval.pipeline.data_loader import load_and_ingest
from rlhf_eval.pipeline.quality_pipeline import run_quality_pipeline

engine = get_engine()
Base.metadata.create_all(engine)

with SessionContext(engine) as session:
    load_and_ingest(session, split="train")

run_quality_pipeline(engine)
```

### 2. Export Flagged Indices

After running the quality pipeline, export flagged indices for the Colab experiment:

```bash
python scripts/export_flagged_indices.py
```

This produces `flagged_indices.json` : a list of dataset indices flagged by any detector.

### 3. Reward Model Experiment (Colab)

1. Upload `notebooks/reward_model_experiment.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU**
3. Run all cells —> upload `flagged_indices.json` when prompted
4. The notebook trains two DistilBERT reward models (clean vs. unfiltered) and compares test accuracy

Training config: `distilbert-base-uncased`, max_length=256, batch_size=8, lr=2e-5, 1 epoch. ~25 minutes total on a free T4.

### 4. Local Reward Model Training

```python
from transformers import AutoTokenizer
from rlhf_eval.reward import (
    RewardModel,
    load_from_huggingface,
    train_reward_model,
    evaluate_reward_model,
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load clean dataset (excluding flagged examples)
flagged = {0, 42, 100}  # or load from flagged_indices.json
train_ds = load_from_huggingface(tokenizer, split="train", exclude_indices=flagged)
test_ds = load_from_huggingface(tokenizer, split="test")

model = RewardModel()
train_reward_model(model, train_ds, epochs=1, device="cuda")
results = evaluate_reward_model(model, test_ds, device="cuda")
print(f"Test accuracy: {results['accuracy']:.4f}")
```

## Database Schema

| Table | Purpose |
|---|---|
| `examples` | Raw HH-RLHF data with parsed prompt, chosen, and rejected turns |
| `quality_signals` | Per-example detector scores and flag status |
| `detector_runs` | Run metadata: thresholds, percentiles, statistics |
| `reward_models` | Trained model records and training configs |
| `evaluations` | Evaluation results linked to reward models |

## Detectors

All detectors extend `BaseDetector` and implement `score()` and `score_batch()`. Thresholds are computed from score distributions using configurable percentiles, with overrides for binary detectors.

| Detector | Method | Threshold Strategy |
|---|---|---|
| **Semantic Similarity** | `all-MiniLM-L6-v2` cosine similarity between chosen/rejected | P99.5 (flags near-duplicates) |
| **Readability Mismatch** | Flesch-Kincaid grade level difference via `textstat` | P95 (rejected reads better) |
| **Repetition** | Unique word ratio in chosen response | P1 (lower = more repetitive) |
| **Length Ratio** | Response length / prompt length | P5 (lower = suspiciously short) |
| **Refusal Bias** | Regex pattern matching for refusal phrases | Fixed threshold (binary) |
| **Unsafe Prompt** | Keyword-based toxicity classification | Fixed threshold (binary) |
| **LLM Judge** | GPT-4o-mini scores 3 dimensions (1–5): helpfulness delta, honest preference, label confidence | Fixed threshold = 3 (flags any dim < 3) |

## Tech Stack

- **Python 3.11+** with full type annotations
- **PostgreSQL 16** + **SQLAlchemy 2.0** (sync ORM)
- **Pydantic Settings** for configuration
- **PyTorch** + **Transformers** for reward model training
- **sentence-transformers** for semantic similarity embeddings
- **textstat** for readability scoring
- **HuggingFace Datasets** for data loading
- **OpenAI** (`gpt-4o-mini`) for LLM judge evaluation
- **FastAPI** + **uvicorn** for the service layer (`/docs` auto-generated)
- **Streamlit** for the interactive dashboard
- **Docker Compose** for one-command deployment
- **GitHub Actions** CI (pytest + mypy + ruff)

## How This Generalizes

This system works on **any dataset of preference pairs**, not just HH-RLHF. The pipeline expects (prompt, chosen, rejected) triplets — the same structure used by TL;DR summarization feedback, OpenAssistant, Ultrafeedback, and every major RLHF dataset. To run it on a different dataset, implement a loader that produces that schema and calls `ingest_to_database`. All 7 detectors, the reward model training loop, and the LLM judge run unchanged. The detector thresholds are learned from each dataset's own score distribution via configurable percentiles, so no calibration is needed. The result is a general-purpose data quality harness for any reward modeling pipeline.

## License

MIT
