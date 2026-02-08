"""Preference pair dataset for reward model training."""

from __future__ import annotations

import re
from typing import Any

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# Lightweight parsing (mirrors rlhf_eval.utils.parsing but avoids import dependency for Colab)
_SPEAKER_RE = re.compile(r"^(Human|Assistant):\s*", re.MULTILINE)


def _last_turn(text: str, role: str) -> str:
    """Extract the last turn for a given role from an HH-RLHF conversation string."""
    parts = _SPEAKER_RE.split(text)
    result = ""
    for i in range(1, len(parts), 2):
        if parts[i].lower() == role and i + 1 < len(parts):
            result = parts[i + 1].strip()
    return result


class PreferencePairDataset(Dataset):
    """Dataset of (prompt, chosen, rejected) preference pairs."""

    def __init__(
        self,
        prompts: list[str],
        chosen_responses: list[str],
        rejected_responses: list[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ) -> None:
        assert len(prompts) == len(chosen_responses) == len(rejected_responses)
        self.prompts = prompts
        self.chosen_responses = chosen_responses
        self.rejected_responses = rejected_responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prompt = self.prompts[idx]

        chosen_enc = self.tokenizer(
            prompt,
            self.chosen_responses[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            prompt,
            self.rejected_responses[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


def load_from_huggingface(
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_length: int = 256,
    exclude_indices: set[int] | None = None,
) -> PreferencePairDataset:
    """Load HH-RLHF from HuggingFace and build a PreferencePairDataset.

    Args:
        tokenizer: Tokenizer for encoding text pairs.
        split: HuggingFace dataset split ("train" or "test").
        max_length: Max token length for encoding.
        exclude_indices: Set of dataset indices to exclude (flagged examples).
    """
    from datasets import load_dataset

    ds = load_dataset("Anthropic/hh-rlhf", split=split)

    prompts, chosen_responses, rejected_responses = [], [], []
    for i, row in enumerate(ds):
        if exclude_indices and i in exclude_indices:
            continue
        prompt = _last_turn(row["chosen"], "human")
        chosen = _last_turn(row["chosen"], "assistant")
        rejected = _last_turn(row["rejected"], "assistant")
        if prompt and chosen and rejected:
            prompts.append(prompt)
            chosen_responses.append(chosen)
            rejected_responses.append(rejected)

    return PreferencePairDataset(
        prompts, chosen_responses, rejected_responses, tokenizer, max_length
    )


def load_from_database(
    tokenizer: PreTrainedTokenizer,
    database_url: str,
    example_ids: list[int] | None = None,
    max_length: int = 256,
) -> PreferencePairDataset:
    """Load preference pairs from PostgreSQL database.

    Args:
        tokenizer: Tokenizer for encoding text pairs.
        database_url: SQLAlchemy database URL.
        example_ids: Optional list of example IDs to include. If None, loads all.
        max_length: Max token length for encoding.
    """
    from sqlalchemy import select

    from rlhf_eval.database.connection import get_engine, get_session
    from rlhf_eval.database.models import Example

    engine = get_engine(database_url)
    session = get_session(engine)

    try:
        query = select(Example)
        if example_ids is not None:
            query = query.where(Example.id.in_(example_ids))

        examples = list(session.execute(query).scalars().all())

        prompts = [ex.prompt for ex in examples]
        chosen_responses = [ex.chosen_last_assistant for ex in examples]
        rejected_responses = [ex.rejected_last_assistant for ex in examples]
    finally:
        session.close()

    return PreferencePairDataset(
        prompts, chosen_responses, rejected_responses, tokenizer, max_length
    )
