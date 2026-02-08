#!/usr/bin/env python3
"""
Quality pipeline for running detectors on ingested examples.

Orchestrates detector execution, stores quality signals, computes
thresholds, and generates quality reports.
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

# ----------------- Third Party Library -----------------
import numpy as np
from tqdm import tqdm

# ----------------- Application Imports -----------------
from rlhf_eval.config.settings import DetectorConfig, get_settings

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from rlhf_eval.database.models import DetectorRun
    from rlhf_eval.detectors.base import BaseDetector

# ----------------- Module-level Configuration -----------------

logger = logging.getLogger(__name__)


def run_detector(
    session: Session,
    detector: BaseDetector,
    threshold_percentile: float = 99.5,
    block_size: int = 1000,
    resume: bool = True,
) -> DetectorRun:
    """
    Run a single detector on all examples in the database.

    Loads examples in chunks, scores using the detector, stores a
    QualitySignal for each example, computes a threshold from the
    given percentile, flags examples above threshold, and records
    a DetectorRun with aggregate statistics.

    Args:
        session: Database session.
        detector: Detector instance implementing BaseDetector.
        threshold_percentile: Percentile for flagging threshold (0-100).
        block_size: Number of examples to process per chunk.
        resume: If True, skip examples that already have signals for this detector.

    Returns:
        DetectorRun with stats and results.
    """
    from rlhf_eval.database.operations import (
        get_example_count,
        get_examples_paginated,
        insert_detector_run,
        insert_quality_signals_batch,
        signal_exists,
        update_detector_run,
    )
    from rlhf_eval.utils.stats import compute_stats

    started_at = datetime.now(timezone.utc)
    logger.info("Starting detector '%s'...", detector.name)

    # Create a detector run record
    run = insert_detector_run(
        session,
        detector_name=detector.name,
        threshold=0.0,  # placeholder — updated after scoring
        percentile_used=threshold_percentile,
        total_examples=0,
        flagged_count=0,
        stats={},
        started_at=started_at,
    )
    session.commit()

    # Collect all scores for threshold computation
    all_scores: list[float] = []
    # Track (example_id, score, metadata) for deferred flagging
    pending_signals: list[tuple[int, float, dict]] = []

    total_examples = get_example_count(session)
    offset = 0

    with tqdm(total=total_examples, desc=f"Scoring [{detector.name}]", unit="ex") as pbar:
        while offset < total_examples:
            examples = get_examples_paginated(session, offset=offset, limit=block_size)
            if not examples:
                break

            # Filter already-processed examples if resuming
            to_process = []
            for ex in examples:
                if resume and signal_exists(session, ex.id, detector.name):
                    pbar.update(1)
                    continue
                to_process.append(ex)

            if to_process:
                # Build pairs for batch scoring.
                # unsafe_prompt scores the human prompt, not assistant responses.
                if detector.name == "unsafe_prompt":
                    pairs = [(ex.prompt, "") for ex in to_process]
                else:
                    pairs = [
                        (ex.chosen_last_assistant, ex.rejected_last_assistant)
                        for ex in to_process
                    ]

                scores = detector.score_batch(pairs, batch_size=get_settings().batch_size)

                for ex, score in zip(to_process, scores):
                    score_val = float(score)
                    if detector.name == "unsafe_prompt":
                        metadata = detector.get_metadata(ex.prompt, "")
                    else:
                        metadata = detector.get_metadata(
                            ex.chosen_last_assistant, ex.rejected_last_assistant
                        )
                    all_scores.append(score_val)
                    pending_signals.append((ex.id, score_val, metadata))
                    pbar.update(1)
            else:
                pbar.update(len(examples))

            offset += block_size

    if not all_scores:
        logger.warning("No new scores computed for detector '%s'.", detector.name)
        update_detector_run(
            session,
            run.id,
            total_examples=0,
            flagged_count=0,
            stats={},
            completed_at=datetime.now(timezone.utc),
        )
        session.commit()
        return run

    # Compute threshold and stats
    scores_array = np.array(all_scores)
    stats = compute_stats(scores_array)

    # Binary/discrete detectors provide a fixed threshold instead of
    # computing one from a percentile (which can misfire on 0/1 scores).
    fixed = detector.get_fixed_threshold()
    if fixed is not None:
        threshold = fixed
        logger.info(
            "Detector '%s': using fixed threshold=%.6f, mean=%.4f, std=%.4f",
            detector.name,
            threshold,
            stats["mean"],
            stats["std"],
        )
    else:
        threshold = float(np.percentile(scores_array, threshold_percentile))
        logger.info(
            "Detector '%s': threshold=%.6f (P%.1f), mean=%.4f, std=%.4f",
            detector.name,
            threshold,
            threshold_percentile,
            stats["mean"],
            stats["std"],
        )

    # Store signals with flagging
    flagged_count = 0
    for block_start in range(0, len(pending_signals), block_size):
        block = pending_signals[block_start : block_start + block_size]
        batch = []
        for example_id, score_val, metadata in block:
            flagged = detector.is_flagged(score_val, threshold)
            if flagged:
                flagged_count += 1
            batch.append(
                {
                    "example_id": example_id,
                    "detector_name": detector.name,
                    "score": score_val,
                    "flagged": flagged,
                    "signal_metadata": metadata if metadata else None,
                }
            )
        insert_quality_signals_batch(session, batch)
        session.commit()

    # Update detector run with final stats
    update_detector_run(
        session,
        run.id,
        threshold=threshold,
        total_examples=len(all_scores),
        flagged_count=flagged_count,
        stats=dict(stats),
        completed_at=datetime.now(timezone.utc),
    )
    session.commit()

    logger.info(
        "Detector '%s' complete: %d scored, %d flagged (%.2f%%).",
        detector.name,
        len(all_scores),
        flagged_count,
        100 * flagged_count / len(all_scores) if all_scores else 0,
    )

    return run


def run_all_detectors(
    session: Session,
    detectors: list[BaseDetector] | None = None,
    block_size: int = 1000,
) -> list[DetectorRun]:
    """
    Run all enabled detectors sequentially.

    If detectors is None, uses all registered detectors that are
    enabled in settings.

    Args:
        session: Database session.
        detectors: Explicit list of detectors, or None for all enabled.
        block_size: Chunk size for processing.

    Returns:
        List of DetectorRun records.
    """
    settings = get_settings()

    if detectors is None:
        detectors = _get_default_detectors(settings)

    runs: list[DetectorRun] = []
    for detector in detectors:
        config = settings.get_detector_config(detector.name)
        if not config.enabled:
            logger.info("Skipping disabled detector '%s'.", detector.name)
            continue

        run = run_detector(
            session,
            detector,
            threshold_percentile=config.threshold_percentile,
            block_size=block_size,
        )
        runs.append(run)

    return runs


def get_clean_example_ids(
    session: Session,
    exclude_detectors: list[str] | None = None,
) -> list[int]:
    """
    Get example IDs that weren't flagged by any detector.

    Args:
        session: Database session.
        exclude_detectors: Optionally ignore flags from these detectors.

    Returns:
        List of example IDs that passed all detectors.
    """
    from rlhf_eval.database.operations import (
        get_all_example_ids,
        get_flagged_example_ids,
        get_latest_detector_run,
    )

    all_ids = set(get_all_example_ids(session))

    # Collect all flagged IDs across detectors
    flagged_ids: set[int] = set()

    # Find all detectors that have run
    settings = get_settings()
    for detector_name in settings.detectors:
        if exclude_detectors and detector_name in exclude_detectors:
            continue

        latest_run = get_latest_detector_run(session, detector_name)
        if latest_run is None:
            continue

        flagged = get_flagged_example_ids(session, detector_name)
        flagged_ids.update(flagged)

    clean_ids = sorted(all_ids - flagged_ids)
    logger.info(
        "Clean examples: %d / %d (excluded %d flagged).",
        len(clean_ids),
        len(all_ids),
        len(flagged_ids),
    )
    return clean_ids


def generate_quality_report(session: Session) -> dict:
    """
    Generate summary report of all detector runs.

    Returns:
        Dict with keys: total_examples, detectors (per-detector stats),
        clean_examples, flagged_any.
    """
    from rlhf_eval.database.operations import (
        get_example_count,
        get_flagged_example_ids,
        get_latest_detector_run,
    )

    settings = get_settings()
    total_examples = get_example_count(session)

    detector_reports: dict[str, dict] = {}
    all_flagged: set[int] = set()

    for detector_name in settings.detectors:
        latest_run = get_latest_detector_run(session, detector_name)
        if latest_run is None:
            detector_reports[detector_name] = {"status": "not_run"}
            continue

        flagged_ids = get_flagged_example_ids(session, detector_name)
        all_flagged.update(flagged_ids)

        detector_reports[detector_name] = {
            "status": "complete",
            "threshold": latest_run.threshold,
            "percentile_used": latest_run.percentile_used,
            "total_scored": latest_run.total_examples,
            "flagged": latest_run.flagged_count,
            "flagged_pct": (
                100 * latest_run.flagged_count / latest_run.total_examples
                if latest_run.total_examples > 0
                else 0.0
            ),
            "stats": latest_run.stats,
            "completed_at": (
                latest_run.completed_at.isoformat()
                if latest_run.completed_at
                else None
            ),
        }

    clean_count = total_examples - len(all_flagged)

    report = {
        "total_examples": total_examples,
        "detectors": detector_reports,
        "clean_examples": clean_count,
        "flagged_any": len(all_flagged),
    }

    logger.info(
        "Quality report: %d total, %d clean, %d flagged by any detector.",
        total_examples,
        clean_count,
        len(all_flagged),
    )
    return report


def _get_default_detectors(settings) -> list[BaseDetector]:
    """
    Instantiate all enabled detectors using the DETECTOR_REGISTRY.

    Uses the registry provided by the detectors package for auto-discovery.
    SemanticSimilarityDetector receives the embedding model name from settings;
    all other detectors are instantiated with defaults.
    """
    try:
        from rlhf_eval.detectors import DETECTOR_REGISTRY
    except ImportError:
        logger.warning("Could not import DETECTOR_REGISTRY; no detectors available.")
        return []

    detectors: list[BaseDetector] = []

    for name, detector_cls in DETECTOR_REGISTRY.items():
        config = settings.detectors.get(name, DetectorConfig())
        if not config.enabled:
            continue

        try:
            if name == "semantic_similarity":
                instance = detector_cls(model_name=settings.embedding_model)
            else:
                instance = detector_cls()
            detectors.append(instance)
        except Exception as exc:
            logger.warning("Could not instantiate detector '%s': %s", name, exc)

    return detectors
