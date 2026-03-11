#!/usr/bin/env python3
"""
LLM Judge Detector.

Uses GPT-4o-mini to evaluate whether human preference labels in RLHF
training data are trustworthy. Scores 3 dimensions on a 1-5 scale:

  - helpfulness_delta: Is chosen meaningfully more helpful?
  - honest_preference: Is chosen better for real vs. surface reasons?
  - label_confidence: How confident is the judge the human got this right?

Flags pairs where any dimension scores below 3. Uses asyncio with a
semaphore to rate-limit concurrent requests to ~50 max.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import numpy as np

from .base import BaseDetector

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are evaluating human preference labels in RLHF training data.
Your job is to flag pairs where the human label is likely wrong,
ambiguous, or reflects bias rather than genuine quality.
Respond in JSON only. No preamble."""

USER_PROMPT_TEMPLATE = """Prompt: {prompt}
Chosen: {chosen}
Rejected: {rejected}

Score each dimension 1-5:
- helpfulness_delta: Is chosen meaningfully more helpful?
- honest_preference: Is chosen better for real reasons vs. surface reasons?
- label_confidence: How confident are you the human label is correct?

Return: {{"helpfulness_delta": int, "honest_preference": int, "label_confidence": int}}"""

FLAG_THRESHOLD = 3.0
DEFAULT_MAX_CONCURRENT = 50
DEFAULT_MODEL = "gpt-4o-mini"


class LLMJudgeDetector(BaseDetector):
    """Use GPT-4o-mini to evaluate trustworthiness of human preference labels.

    Scores 3 dimensions (1-5 each). The returned score is the minimum across
    all 3 dimensions. A score below 3 indicates at least one dimension flagged.

    This detector is designed for batch async processing. Do NOT call it
    synchronously on large datasets without rate-limiting — use score_batch
    which handles concurrency automatically.

    Estimated cost on full HH-RLHF (160K pairs): ~$15-25 with gpt-4o-mini.
    """

    name = "llm_judge"
    description = "LLM-based evaluation of human preference label trustworthiness"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> None:
        """
        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: OpenAI model to use for judging.
            max_concurrent: Maximum number of concurrent API requests.
        """
        import openai

        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model
        self._max_concurrent = max_concurrent
        # Cache populated during score_batch, read by get_metadata
        self._metadata_cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _judge_single(
        self,
        semaphore: asyncio.Semaphore,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> dict[str, Any]:
        """Call GPT-4o-mini for one preference pair."""
        async with semaphore:
            user_msg = USER_PROMPT_TEMPLATE.format(
                prompt=prompt[:2000],
                chosen=chosen[:2000],
                rejected=rejected[:2000],
            )
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=100,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or "{}"
                scores = json.loads(raw)
                helpfulness = int(scores.get("helpfulness_delta", 3))
                honest = int(scores.get("honest_preference", 3))
                confidence = int(scores.get("label_confidence", 3))
                return {
                    "helpfulness_delta": helpfulness,
                    "honest_preference": honest,
                    "label_confidence": confidence,
                    "min_score": min(helpfulness, honest, confidence),
                    "judge_flagged": min(helpfulness, honest, confidence) < FLAG_THRESHOLD,
                }
            except Exception as exc:
                logger.warning("LLM judge failed for pair: %s", exc)
                return {
                    "helpfulness_delta": 3,
                    "honest_preference": 3,
                    "label_confidence": 3,
                    "min_score": 3.0,
                    "judge_flagged": False,
                    "error": str(exc),
                }

    async def _judge_batch_async(
        self,
        triplets: list[tuple[str, str, str]],
    ) -> list[dict[str, Any]]:
        """Run all pairs concurrently with semaphore rate limiting."""
        semaphore = asyncio.Semaphore(self._max_concurrent)
        tasks = [
            self._judge_single(semaphore, prompt, chosen, rejected)
            for prompt, chosen, rejected in triplets
        ]
        return await asyncio.gather(*tasks)

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def score(self, chosen_text: str, rejected_text: str) -> float:
        """Score a single pair synchronously.

        For the LLM judge, chosen_text should be the prompt and
        rejected_text should be the chosen response (use score_batch
        for proper triplet handling in the pipeline).
        """
        # Minimal single-pair implementation for API/test usage
        result = asyncio.run(
            self._judge_single(
                asyncio.Semaphore(1),
                chosen_text,  # treated as prompt
                rejected_text,  # treated as chosen
                "",
            )
        )
        return float(result["min_score"])

    def score_batch(
        self,
        pairs: list[tuple],  # type: ignore[override]
        _batch_size: int = 32,
    ) -> np.ndarray:
        """Score a batch of preference pairs.

        Args:
            pairs: List of (prompt, chosen, rejected) triplets.
                   The pipeline passes these via a special case for "llm_judge".
            _batch_size: Unused — concurrency controlled by max_concurrent.

        Returns:
            Array of min-dimension scores (1.0-5.0). Scores < 3.0 are flagged.
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        logger.info("LLM judge: scoring %d pairs (max_concurrent=%d)", len(pairs), self._max_concurrent)

        # Run async batch
        results = asyncio.run(self._judge_batch_async(pairs))

        # Populate metadata cache and collect scores
        self._metadata_cache = {}
        scores: list[float] = []

        for triplet, result in zip(pairs, results):
            prompt = triplet[0] if len(triplet) > 0 else ""
            chosen = triplet[1] if len(triplet) > 1 else ""
            cache_key = self._cache_key(prompt, chosen)
            self._metadata_cache[cache_key] = result
            scores.append(float(result["min_score"]))

        flagged = sum(1 for s in scores if s < FLAG_THRESHOLD)
        logger.info(
            "LLM judge: %d scored, %d flagged (%.1f%%)",
            len(scores),
            flagged,
            100 * flagged / len(scores) if scores else 0,
        )

        return np.array(scores, dtype=np.float32)

    def is_flagged(self, score: float, threshold: float) -> bool:
        """Flag when the minimum dimension score is below threshold."""
        return bool(score < threshold)

    def get_fixed_threshold(self) -> float:
        """Always use 3.0 — flag when any dimension < 3."""
        return FLAG_THRESHOLD

    def get_default_threshold_percentile(self) -> float:
        """Not used (fixed threshold), but required by interface."""
        return 99.5

    def get_metadata(self, prompt: str, chosen_text: str) -> dict[str, Any]:
        """Return cached judge scores for this pair.

        Called by the pipeline after score_batch completes.
        The pipeline passes (ex.prompt, ex.chosen_last_assistant).
        """
        cache_key = self._cache_key(prompt, chosen_text)
        return self._metadata_cache.get(cache_key, {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(prompt: str, chosen: str) -> str:
        return f"{prompt[:80]}|||{chosen[:80]}"
