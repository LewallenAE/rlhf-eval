# RLHF Evaluation Harness

An end-to-end system for detecting problematic preference pairs in RLHF training data, training reward models on filtered vs. unfiltered datasets, and measuring the impact of data quality on model performance.

Built against Anthropic's [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset (160,800 preference pairs).

## Motivation

Reward models are only as good as the preference data they're trained on. Noisy labels, near-duplicate responses, degenerate text, and labeling bias all degrade reward signal quality. This project identifies and removes those failure modes, then empirically validates the improvement by training competing reward models.

## Results

**Data Quality Pipeline** — 160,800 examples scored by 6 detectors:

| Detector | Flagged | Rate | What It Catches |
|---|---|---|---|
| Semantic Similarity | 804 | 0.5% | Near-identical chosen/rejected pairs |
| Readability Mismatch | 8,041 | 5.0% | Rejected response has better readability |
| Repetition | 1,610 | 1.0% | Degenerate repeated text |
| Length Ratio | 8,067 | 5.0% | Suspiciously short responses to complex prompts |
| Refusal Bias | 291 | 0.2% | Chosen refuses while rejected is helpful |
| Unsafe Prompt | 1,680 | 1.0% | Prompts where neither response may be valid |
| **Total unique** | **12,693** | **7.9%** | |

**148,107 clean examples** retained (92.1% of dataset).

**Reward Model Comparison** — DistilBERT (67M params), Bradley-Terry loss, 1 epoch on T4 GPU:

| Metric | Clean (filtered) | Unfiltered |
|---|---|---|
| Training examples | 148,107 | 160,800 |
| Test accuracy | TBD | TBD |

*Test accuracy populated after running the Colab experiment notebook.*

![Reward Model Comparison](src/results/Preference_Pair_Test_Result_01.webp)
![Reward Model Comparison2](src/results/Prefernce_Pair_Test_Result_02.webp)

## Architecture

```
src/rlhf_eval/
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
│   └── unsafe_prompt.py       # Toxicity keyword classification
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

scripts/
└── export_flagged_indices.py  # Export flagged indices for Colab

notebooks/
└── reward_model_experiment.ipynb  # Self-contained Colab notebook

tests/                         # 207 tests across 4 test modules
```

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 16 (Docker recommended)

### Database

```bash
docker run -d --name rlhf-postgres \
  -p 5432:5432 \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=pass \
  -e POSTGRES_DB=rlhf_dev \
  -v rlhf_pgdata:/var/lib/postgresql/data \
  postgres:16
```

### Installation

```bash
git clone https://github.com/LewallenAE/rlhf-eval.git
cd rlhf-eval
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Environment

Create a `.env` file in the project root:

```
RLHF_DATABASE_URL=postgresql://user:pass@localhost:5432/rlhf_dev
```

### Run Tests

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

This produces `flagged_indices.json` — a list of dataset indices flagged by any detector.

### 3. Reward Model Experiment (Colab)

1. Upload `notebooks/reward_model_experiment.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU**
3. Run all cells — upload `flagged_indices.json` when prompted
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

## Tech Stack

- **Python 3.11+** with full type annotations
- **PostgreSQL 16** + **SQLAlchemy 2.0** (sync ORM)
- **Pydantic Settings** for configuration
- **PyTorch** + **Transformers** for reward model training
- **sentence-transformers** for semantic similarity embeddings
- **textstat** for readability scoring
- **HuggingFace Datasets** for data loading
- **pytest** (207 tests)

## License

MIT
