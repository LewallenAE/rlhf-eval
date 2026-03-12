#!/usr/bin/env python3
"""
FastAPI service layer for the RLHF Evaluation Harness.

Endpoints:
  POST /ingest      — ingest HH-RLHF or custom CSV/JSON data
  POST /score       — run quality detectors on ingested data
  GET  /experiments — return clean vs. unfiltered comparison metrics

Visit /docs for auto-generated Swagger UI.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Lifespan: create tables on startup ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from rlhf_eval.database.connection import get_engine
        from rlhf_eval.database.models import Base
        engine = get_engine()
        Base.metadata.create_all(engine)
        logger.info("Database tables created/verified.")
    except Exception as exc:
        logger.warning("Could not connect to database on startup: %s", exc)
    yield


app = FastAPI(
    title="RLHF Evaluation Harness",
    description=(
        "API for ingesting preference pair data, running quality detectors, "
        "and comparing clean vs. unfiltered reward model training results."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request / Response models ───────────────────────────────────────

class IngestRequest(BaseModel):
    source: str = "hh-rlhf"
    split: str = "train"
    limit: int | None = None


class IngestResponse(BaseModel):
    total_processed: int
    total_inserted: int
    skipped: int
    errors: int


class ScoreRequest(BaseModel):
    detector_names: list[str] = ["all"]
    block_size: int = 1000


class ScoreResponse(BaseModel):
    detectors_run: list[str]
    total_flagged: int
    report: dict[str, Any]


class ExperimentsResponse(BaseModel):
    total_examples: int
    clean_examples: int
    flagged_any: int
    clean_pct: float
    detectors: dict[str, Any]


class ReviewRequest(BaseModel):
    example_index: int
    detector_name: str
    label: str  # true_positive | false_positive | needs_review


class ReviewResponse(BaseModel):
    example_index: int
    detector_name: str
    label: str


# ── Endpoints ───────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, summary="Ingest preference pair data")
def ingest(request: IngestRequest) -> IngestResponse:
    """Load HH-RLHF from HuggingFace and ingest into the database.

    - **source**: Dataset source. Currently supports `hh-rlhf`.
    - **split**: Dataset split (`train` or `test`).
    - **limit**: Optional limit on number of examples (for testing).
    """
    try:
        from rlhf_eval.database.connection import get_engine, SessionContext
        from rlhf_eval.pipeline.data_loader import load_hh_rlhf, ingest_to_database

        if request.source != "hh-rlhf":
            raise HTTPException(status_code=400, detail=f"Unsupported source: {request.source!r}. Use 'hh-rlhf'.")

        dataset = load_hh_rlhf(split=request.split)

        if request.limit is not None:
            dataset = dataset.select(range(min(request.limit, len(dataset))))

        engine = get_engine()
        with SessionContext(engine) as session:
            result = ingest_to_database(session, dataset)

        return IngestResponse(
            total_processed=result["total_processed"],
            total_inserted=result["total_inserted"],
            skipped=result["skipped"],
            errors=len(result["errors"]),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/score", response_model=ScoreResponse, summary="Run quality detectors")
def score(request: ScoreRequest) -> ScoreResponse:
    """Run quality detectors on all ingested examples.

    - **detector_names**: List of detector names to run, or `["all"]` for all enabled detectors.
    - **block_size**: Processing chunk size (default 1000).

    Returns a summary of which detectors ran and how many examples were flagged.
    """
    try:
        from rlhf_eval.database.connection import get_engine, SessionContext
        from rlhf_eval.pipeline.quality_pipeline import (
            run_all_detectors,
            generate_quality_report,
        )

        engine = get_engine()
        with SessionContext(engine) as session:
            if request.detector_names == ["all"]:
                runs = run_all_detectors(session, block_size=request.block_size)
            else:
                # Import only requested detectors
                from rlhf_eval.detectors import DETECTOR_REGISTRY
                detectors = []
                for name in request.detector_names:
                    if name not in DETECTOR_REGISTRY:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unknown detector: {name!r}. Available: {list(DETECTOR_REGISTRY)}",
                        )
                    detectors.append(DETECTOR_REGISTRY[name]())
                runs = run_all_detectors(session, detectors=detectors, block_size=request.block_size)

            report = generate_quality_report(session)
            total_flagged = report.get("flagged_any", 0)
            detectors_run = [run.detector_name for run in runs]

        return ScoreResponse(
            detectors_run=detectors_run,
            total_flagged=total_flagged,
            report=report,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/experiments", response_model=ExperimentsResponse, summary="Quality experiment results")
def experiments() -> ExperimentsResponse:
    """Return quality pipeline results: clean vs. flagged example counts per detector.

    Use this to compare how many examples survived filtering vs. were flagged,
    and which detectors caught the most problematic pairs.
    """
    try:
        from rlhf_eval.database.connection import get_engine, SessionContext
        from rlhf_eval.pipeline.quality_pipeline import generate_quality_report

        engine = get_engine()
        with SessionContext(engine) as session:
            report = generate_quality_report(session)

        total = report["total_examples"]
        clean = report["clean_examples"]

        return ExperimentsResponse(
            total_examples=total,
            clean_examples=clean,
            flagged_any=report["flagged_any"],
            clean_pct=round(100 * clean / total, 2) if total > 0 else 0.0,
            detectors=report["detectors"],
        )

    except Exception as exc:
        logger.exception("Experiments query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reviews", response_model=ReviewResponse, summary="Submit a reviewer label")
def submit_review(request: ReviewRequest) -> ReviewResponse:
    """Label a flagged example as true_positive, false_positive, or needs_review.

    - **example_index**: The dataset_index of the example to review.
    - **detector_name**: Which detector's flag you are reviewing.
    - **label**: One of `true_positive`, `false_positive`, `needs_review`.
    """
    valid_labels = {"true_positive", "false_positive", "needs_review"}
    if request.label not in valid_labels:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid label {request.label!r}. Must be one of {sorted(valid_labels)}.",
        )

    try:
        from rlhf_eval.database.connection import get_engine, SessionContext
        from rlhf_eval.database.operations import get_example_by_index, upsert_review

        engine = get_engine()
        with SessionContext(engine) as session:
            example = get_example_by_index(session, request.example_index)
            if example is None:
                raise HTTPException(status_code=404, detail=f"Example index {request.example_index} not found.")
            upsert_review(session, example.id, request.detector_name, request.label)
            session.commit()

        return ReviewResponse(
            example_index=request.example_index,
            detector_name=request.detector_name,
            label=request.label,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Review submission failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/reviews/stats", summary="Per-detector reviewer precision stats")
def review_stats() -> dict[str, Any]:
    """Return TP/FP/needs_review counts per detector from human reviews."""
    try:
        from rlhf_eval.database.connection import get_engine, SessionContext
        from rlhf_eval.database.operations import get_review_stats

        engine = get_engine()
        with SessionContext(engine) as session:
            return get_review_stats(session)

    except Exception as exc:
        logger.exception("Review stats query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", summary="Health check")
def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}
