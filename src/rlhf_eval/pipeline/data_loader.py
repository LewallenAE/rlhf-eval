#!/usr/bin/env python3
"""
Data loader for the Anthropic HH-RLHF dataset.

Handles loading from HuggingFace, parsing conversations,
and ingesting into the database with chunked processing and resume support.
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
import logging
from typing import TYPE_CHECKING

# ----------------- Third Party Library -----------------
from tqdm import tqdm

# ----------------- Application Imports -----------------
from rlhf_eval.utils.parsing import last_assistant_response, last_human_prompt

if TYPE_CHECKING:
    from datasets import Dataset
    from sqlalchemy.orm import Session

# ----------------- Module-level Configuration -----------------

logger = logging.getLogger(__name__)

DATASET_NAME = "Anthropic/hh-rlhf"


def load_hh_rlhf(split: str = "train") -> Dataset:
    """
    Load HH-RLHF dataset from HuggingFace.

    Args:
        split: "train" or "test"

    Returns:
        HuggingFace Dataset object with 'chosen' and 'rejected' columns.
    """
    from datasets import load_dataset

    logger.info("Loading HH-RLHF dataset (split=%s)...", split)
    dataset = load_dataset(DATASET_NAME, split=split)
    logger.info("Loaded %d examples.", len(dataset))
    return dataset


def ingest_to_database(
    session: Session,
    dataset: Dataset,
    block_size: int = 1000,
    resume_from: int = 0,
) -> dict:
    """
    Parse all examples and store to database.

    Extracts last assistant turn and last human prompt from each
    conversation using parsing utilities, then inserts in batches.
    Supports resuming from a specific dataset index.

    Args:
        session: SQLAlchemy session.
        dataset: HuggingFace Dataset with 'chosen' and 'rejected' columns.
        block_size: Number of examples to process per batch.
        resume_from: Dataset index to resume from (skip earlier indices).

    Returns:
        Dict with keys: total_processed, total_inserted, skipped, errors.
    """
    from rlhf_eval.database.operations import (
        get_example_by_index,
        insert_examples_batch,
    )

    total_processed = 0
    total_inserted = 0
    skipped = 0
    errors: list[dict] = []

    total = len(dataset)
    logger.info(
        "Ingesting %d examples (resume_from=%d, block_size=%d)...",
        total,
        resume_from,
        block_size,
    )

    for block_start in tqdm(
        range(resume_from, total, block_size),
        desc="Ingesting blocks",
        unit="block",
    ):
        block_end = min(block_start + block_size, total)
        batch: list[dict] = []

        for idx in range(block_start, block_end):
            total_processed += 1

            # Skip if already ingested
            existing = get_example_by_index(session, idx)
            if existing is not None:
                skipped += 1
                continue

            row = dataset[idx]
            chosen_full: str = row["chosen"].replace("\x00", "")
            rejected_full: str = row["rejected"].replace("\x00", "")

            try:
                chosen_last = last_assistant_response(chosen_full)
                rejected_last = last_assistant_response(rejected_full)
                prompt = last_human_prompt(chosen_full)

                batch.append(
                    {
                        "dataset_index": idx,
                        "chosen_full": chosen_full,
                        "rejected_full": rejected_full,
                        "chosen_last_assistant": chosen_last,
                        "rejected_last_assistant": rejected_last,
                        "prompt": prompt,
                    }
                )
            except Exception as exc:
                errors.append({"index": idx, "error": str(exc)})
                logger.warning("Error parsing example %d: %s", idx, exc)

        if batch:
            insert_examples_batch(session, batch)
            session.commit()
            total_inserted += len(batch)

        logger.debug(
            "Block %d–%d done: inserted=%d, skipped=%d",
            block_start,
            block_end,
            len(batch),
            block_end - block_start - len(batch),
        )

    result = {
        "total_processed": total_processed,
        "total_inserted": total_inserted,
        "skipped": skipped,
        "errors": errors,
    }
    logger.info(
        "Ingestion complete: processed=%d, inserted=%d, skipped=%d, errors=%d",
        total_processed,
        total_inserted,
        skipped,
        len(errors),
    )
    return result


def get_ingestion_progress(session: Session) -> dict:
    """
    Check how much data has been ingested.

    Returns:
        Dict with keys: count (total examples in DB), max_index (highest dataset_index).
    """
    from rlhf_eval.database.operations import get_example_count

    from sqlalchemy import func, select
    from rlhf_eval.database.models import Example

    count = get_example_count(session)

    stmt = select(func.max(Example.dataset_index))
    max_index = session.execute(stmt).scalar()

    return {
        "count": count,
        "max_index": max_index if max_index is not None else -1,
    }
