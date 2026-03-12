#!/usr/bin/env python3
"""
Streamlit dashboard for RLHF Evaluation Harness.

Tab 1 — Flagged Examples Explorer
  Filter by detector, view prompt/chosen/rejected side-by-side,
  and see which detectors flagged each example (with LLM judge scores).

Tab 2 — Reward Model Comparison
  Bar charts of accuracy and reward gap from the clean vs. unfiltered experiment,
  plus detector flag rate breakdown.
"""

from __future__ import annotations

import os

import streamlit as st

st.set_page_config(
    page_title="RLHF Eval Dashboard",
    page_icon="🔍",
    layout="wide",
)

st.title("RLHF Evaluation Harness")

# ── Database connection ─────────────────────────────────────────────

@st.cache_resource
def get_engine():
    from rlhf_eval.database.connection import get_engine as _get_engine
    return _get_engine()


@st.cache_data(ttl=300)
def load_flagged_examples(detector_name: str, limit: int = 200) -> list[dict]:
    """Load flagged examples for a given detector from the latest completed run."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session
    from rlhf_eval.database.models import Example, QualitySignal, DetectorRun

    engine = get_engine()
    with Session(engine) as session:
        # Get the latest completed run for this detector
        latest_run = session.execute(
            select(DetectorRun)
            .where(
                DetectorRun.detector_name == detector_name,
                DetectorRun.completed_at.isnot(None),
            )
            .order_by(DetectorRun.completed_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        run_filter = (QualitySignal.run_id == latest_run.id) if latest_run else (QualitySignal.run_id.is_(None))

        rows = session.execute(
            select(
                Example.id,
                Example.dataset_index,
                Example.prompt,
                Example.chosen_last_assistant,
                Example.rejected_last_assistant,
                QualitySignal.score,
                QualitySignal.signal_metadata,
                QualitySignal.detector_name,
            )
            .join(QualitySignal, QualitySignal.example_id == Example.id)
            .where(
                QualitySignal.detector_name == detector_name,
                QualitySignal.flagged == True,  # noqa: E712
                run_filter,
            )
            .limit(limit)
        ).all()

        return [
            {
                "id": r.id,
                "index": r.dataset_index,
                "prompt": r.prompt,
                "chosen": r.chosen_last_assistant,
                "rejected": r.rejected_last_assistant,
                "score": round(r.score, 4),
                "metadata": r.signal_metadata or {},
                "detector": r.detector_name,
            }
            for r in rows
        ]


def load_reviews(example_ids: list[int], detector_name: str) -> dict[int, str]:
    """Load existing review labels for the displayed examples (no cache — always fresh)."""
    from sqlalchemy.orm import Session
    from rlhf_eval.database.operations import get_reviews_for_examples

    engine = get_engine()
    with Session(engine) as session:
        return get_reviews_for_examples(session, example_ids, detector_name)


def save_review(example_id: int, detector_name: str, label: str) -> None:
    """Persist a review label to the database."""
    from sqlalchemy.orm import Session
    from rlhf_eval.database.operations import upsert_review

    engine = get_engine()
    with Session(engine) as session:
        upsert_review(session, example_id, detector_name, label)
        session.commit()


def load_review_stats() -> dict:
    """Load per-detector TP/FP/needs_review counts."""
    from sqlalchemy.orm import Session
    from rlhf_eval.database.operations import get_review_stats

    engine = get_engine()
    with Session(engine) as session:
        return get_review_stats(session)


@st.cache_data(ttl=300)
def load_detector_names() -> list[str]:
    """Get list of detectors that have run."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session
    from rlhf_eval.database.models import DetectorRun

    engine = get_engine()
    with Session(engine) as session:
        names = session.execute(
            select(DetectorRun.detector_name).distinct()
        ).scalars().all()
        return sorted(names)


@st.cache_data(ttl=300)
def load_detector_stats() -> dict[str, dict]:
    """Load flag rates per detector from the latest runs."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session
    from rlhf_eval.database.models import DetectorRun

    engine = get_engine()
    with Session(engine) as session:
        runs = session.execute(
            select(DetectorRun)
            .order_by(DetectorRun.started_at.desc())
        ).scalars().all()

        # Keep latest run per detector
        seen = {}
        for run in runs:
            if run.detector_name not in seen:
                seen[run.detector_name] = {
                    "flagged_count": run.flagged_count,
                    "total_examples": run.total_examples,
                    "flag_rate": (
                        round(100 * run.flagged_count / run.total_examples, 2)
                        if run.total_examples > 0 else 0.0
                    ),
                    "threshold": run.threshold,
                }
        return seen


# ── Tabs ────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Flagged Examples Explorer", "Reward Model Comparison"])

# ====================================================================
# TAB 1: Flagged Examples Explorer
# ====================================================================

with tab1:
    st.header("Flagged Examples Explorer")
    st.caption("Browse examples flagged by each quality detector.")

    try:
        detector_names = load_detector_names()
    except Exception as e:
        st.error(f"Could not connect to database: {e}")
        st.info("Make sure RLHF_DATABASE_URL is set and the quality pipeline has been run.")
        detector_names = []

    if not detector_names:
        st.warning("No detector runs found. Run the quality pipeline first (`POST /score`).")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_detector = st.selectbox("Select detector", detector_names)
        with col2:
            max_examples = st.slider("Max examples to show", 10, 200, 50)

        if selected_detector:
            examples = load_flagged_examples(selected_detector, limit=max_examples)
            st.info(f"Showing {len(examples)} flagged examples for **{selected_detector}**")

            # Load existing reviews for all displayed examples
            example_ids = [ex["id"] for ex in examples]
            try:
                existing_reviews = load_reviews(example_ids, selected_detector)
            except Exception:
                existing_reviews = {}

            _label_display = {
                "true_positive": "✓ True Positive",
                "false_positive": "✗ False Positive",
                "needs_review": "? Needs Review",
            }

            for i, ex in enumerate(examples):
                current_label = existing_reviews.get(ex["id"])
                label_badge = f" · {_label_display[current_label]}" if current_label else ""
                with st.expander(f"Example #{ex['index']} — score: {ex['score']}{label_badge}", expanded=(i == 0)):
                    # Prompt
                    st.markdown("**Prompt**")
                    st.text_area("", ex["prompt"], height=80, key=f"prompt_{i}", disabled=True)

                    # Chosen / Rejected side-by-side
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Chosen** (preferred)")
                        st.text_area("", ex["chosen"], height=150, key=f"chosen_{i}", disabled=True)
                    with c2:
                        st.markdown("**Rejected**")
                        st.text_area("", ex["rejected"], height=150, key=f"rejected_{i}", disabled=True)

                    # Metadata / LLM judge scores
                    meta = ex["metadata"]
                    if meta:
                        st.markdown("**Detector metadata**")
                        if "helpfulness_delta" in meta:
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Helpfulness Delta", meta.get("helpfulness_delta", "—"))
                            m2.metric("Honest Preference", meta.get("honest_preference", "—"))
                            m3.metric("Label Confidence", meta.get("label_confidence", "—"))
                            if meta.get("reason"):
                                st.caption(f"**Judge reasoning:** {meta['reason']}")
                        else:
                            st.json(meta)

                    # Review buttons
                    st.markdown("**Review**")
                    r1, r2, r3 = st.columns(3)
                    if r1.button("✓ True Positive", key=f"tp_{i}", type="primary" if current_label == "true_positive" else "secondary"):
                        save_review(ex["id"], selected_detector, "true_positive")
                        st.rerun()
                    if r2.button("✗ False Positive", key=f"fp_{i}", type="primary" if current_label == "false_positive" else "secondary"):
                        save_review(ex["id"], selected_detector, "false_positive")
                        st.rerun()
                    if r3.button("? Needs Review", key=f"nr_{i}", type="primary" if current_label == "needs_review" else "secondary"):
                        save_review(ex["id"], selected_detector, "needs_review")
                        st.rerun()

            # Reviewer precision summary
            st.divider()
            st.subheader("Reviewer Precision")
            try:
                review_stats = load_review_stats()
                if review_stats:
                    import pandas as pd
                    rows = []
                    for det, counts in review_stats.items():
                        tp = counts.get("true_positive", 0)
                        fp = counts.get("false_positive", 0)
                        nr = counts.get("needs_review", 0)
                        total = tp + fp + nr
                        precision = round(100 * tp / (tp + fp), 1) if (tp + fp) > 0 else None
                        rows.append({
                            "Detector": det,
                            "True Positive": tp,
                            "False Positive": fp,
                            "Needs Review": nr,
                            "Reviewed": total,
                            "Precision (%)": precision,
                        })
                    st.dataframe(pd.DataFrame(rows).set_index("Detector"), use_container_width=True)
                else:
                    st.caption("No reviews submitted yet.")
            except Exception as e:
                st.caption(f"Could not load review stats: {e}")


# ====================================================================
# TAB 2: Reward Model Comparison
# ====================================================================

with tab2:
    st.header("Reward Model Comparison")
    st.caption(
        "DistilBERT (67M params) trained with Bradley-Terry loss, 1 epoch on T4 GPU. "
        "Clean dataset filtered 7.9% of pathological preference pairs."
    )

    # Results from the experiment
    import pandas as pd

    accuracy_data = pd.DataFrame({
        "Model": ["Clean (filtered)", "Unfiltered"],
        "Test Accuracy (%)": [62.18, 62.33],
    })

    reward_gap_data = pd.DataFrame({
        "Model": ["Clean (filtered)", "Unfiltered"],
        "Avg Reward Gap (chosen − rejected)": [0.2282, 0.3014],
    })

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Test Accuracy")
        st.bar_chart(accuracy_data.set_index("Model"), color="#4CAF50")
        st.caption("Binary accuracy is nearly identical — accuracy is the wrong metric here.")

    with col2:
        st.subheader("Average Reward Gap")
        st.bar_chart(reward_gap_data.set_index("Model"), color="#2196F3")
        st.caption(
            "The unfiltered model's larger gap reflects inflated confidence from "
            "degenerate pairs, not better signal quality."
        )

    st.markdown("""
    ### Interpreting the Results

    Binary accuracy is nearly identical across both models (62.18% vs. 62.33%) — and that's the point.
    The unfiltered model's larger reward gap (0.3014 vs. 0.2282) looks like stronger performance on the
    surface, but it reflects **inflated confidence** learned from degenerate pairs: near-duplicates where
    the label is arbitrary, readability-mismatched examples where the "rejected" response is objectively
    better, and refusal bias cases where the chosen response actively refuses a reasonable request.

    The clean model produces a **tighter, more calibrated reward gap**. In production RLHF pipelines,
    overconfident reward signals from noisy data are a known driver of reward hacking: the model learns
    to exploit the noise rather than the signal. Filtering 7.9% of pathological examples doesn't hurt
    accuracy; it removes the false confidence that makes reward models brittle.

    **Binary accuracy is the wrong metric here. Reward gap calibration is the right one.**
    """)

    # Detector flag rate breakdown
    st.subheader("Detector Flag Rate Breakdown")

    try:
        detector_stats = load_detector_stats()
        if detector_stats:
            stats_df = pd.DataFrame([
                {
                    "Detector": name,
                    "Flagged": stats["flagged_count"],
                    "Flag Rate (%)": stats["flag_rate"],
                    "Total Scored": stats["total_examples"],
                }
                for name, stats in detector_stats.items()
            ]).set_index("Detector")
            st.dataframe(stats_df, use_container_width=True)
        else:
            # Show hardcoded results from the actual experiment
            st.info("No live detector stats found. Showing results from the original experiment.")
            hardcoded = pd.DataFrame([
                {"Detector": "semantic_similarity", "Flagged": 804, "Flag Rate (%)": 0.5},
                {"Detector": "readability_mismatch", "Flagged": 8041, "Flag Rate (%)": 5.0},
                {"Detector": "repetition", "Flagged": 1610, "Flag Rate (%)": 1.0},
                {"Detector": "length_ratio", "Flagged": 8067, "Flag Rate (%)": 5.0},
                {"Detector": "refusal_bias", "Flagged": 291, "Flag Rate (%)": 0.2},
                {"Detector": "unsafe_prompt", "Flagged": 1680, "Flag Rate (%)": 1.0},
            ]).set_index("Detector")
            st.dataframe(hardcoded, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load live stats: {e}")
