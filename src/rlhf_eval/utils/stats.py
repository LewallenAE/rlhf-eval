#!/usr/bin/env python3
"""
Statistical utilities for detector score analysis.

Provides percentile computation, distribution summaries, threshold cliff
detection, and two-sample distribution comparison (KS test).
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
from typing import TypedDict

# ----------------- Third Party Library -----------------
import numpy as np
from scipy import stats as sp_stats

# ----------------- Application Imports -----------------

# ----------------- Module-level Configuration -----------------


class StatsDict(TypedDict):
    """Comprehensive statistics for an array of values."""

    mean: float
    std: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float
    p99_5: float


def compute_stats(values: np.ndarray) -> StatsDict:
    """
    Compute comprehensive statistics for an array of values.

    Args:
        values: 1-D array of numeric values.

    Returns:
        A ``StatsDict`` with mean, std, min, max, and key percentiles.

    Raises:
        ValueError: If *values* is empty.
    """
    if values.size == 0:
        raise ValueError("Cannot compute stats on an empty array.")

    return StatsDict(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        p50=float(np.percentile(values, 50)),
        p90=float(np.percentile(values, 90)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
        p99_5=float(np.percentile(values, 99.5)),
    )


def compute_percentile(values: np.ndarray, p: float) -> float:
    """
    Compute a single percentile.

    Args:
        values: 1-D array of numeric values.
        p: Percentile to compute (0–100).

    Returns:
        The percentile value as a float.

    Raises:
        ValueError: If *values* is empty or *p* is out of range.
    """
    if values.size == 0:
        raise ValueError("Cannot compute percentile on an empty array.")
    if not 0 <= p <= 100:
        raise ValueError(f"Percentile must be between 0 and 100, got {p}.")

    return float(np.percentile(values, p))


def find_threshold_cliff(
    values: np.ndarray,
    percentiles: list[float] | None = None,
) -> tuple[float, float, float]:
    """
    Find where the distribution jumps most significantly between percentiles.

    Compares consecutive percentile values and returns the pair with the
    largest absolute jump.

    Args:
        values: 1-D array of numeric values.
        percentiles: Percentiles to evaluate.  Defaults to
            ``[90, 95, 99, 99.5, 99.9]``.

    Returns:
        A 3-tuple ``(percentile, threshold_value, jump_size)`` where
        *percentile* is the upper percentile at which the cliff occurs,
        *threshold_value* is the value at that percentile, and
        *jump_size* is the absolute difference from the previous percentile.

    Raises:
        ValueError: If *values* is empty or fewer than 2 percentiles given.
    """
    if values.size == 0:
        raise ValueError("Cannot find threshold cliff on an empty array.")

    if percentiles is None:
        percentiles = [90.0, 95.0, 99.0, 99.5, 99.9]

    if len(percentiles) < 2:
        raise ValueError("Need at least 2 percentiles to find a cliff.")

    pct_values = [float(np.percentile(values, p)) for p in percentiles]

    max_jump = 0.0
    cliff_idx = 1
    for i in range(1, len(pct_values)):
        jump = abs(pct_values[i] - pct_values[i - 1])
        if jump > max_jump:
            max_jump = jump
            cliff_idx = i

    return (percentiles[cliff_idx], pct_values[cliff_idx], max_jump)


def compare_distributions(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> dict[str, float]:
    """
    Compare two distributions using the two-sample KS test.

    Useful for drift detection — e.g. comparing detector score distributions
    across different runs or dataset slices.

    Args:
        values_a: First sample (1-D array).
        values_b: Second sample (1-D array).

    Returns:
        Dictionary with keys:
        - ``ks_statistic``: KS test statistic.
        - ``p_value``: Associated p-value.
        - ``mean_diff``: Difference in means (a − b).
        - ``std_diff``: Difference in standard deviations (a − b).

    Raises:
        ValueError: If either array is empty.
    """
    if values_a.size == 0 or values_b.size == 0:
        raise ValueError("Cannot compare distributions with an empty array.")

    ks_stat, p_value = sp_stats.ks_2samp(values_a, values_b)

    return {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "mean_diff": float(np.mean(values_a) - np.mean(values_b)),
        "std_diff": float(np.std(values_a) - np.std(values_b)),
    }
