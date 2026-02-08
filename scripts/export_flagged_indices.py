#!/usr/bin/env python3
"""Export flagged dataset indices to JSON for use in Colab."""

import json

from sqlalchemy import select

from rlhf_eval.database.connection import get_engine, get_session
from rlhf_eval.database.models import Example, QualitySignal


def main() -> None:
    engine = get_engine()
    session = get_session(engine)

    # Get dataset_index for any example flagged by at least one detector
    flagged_indices = list(
        session.execute(
            select(Example.dataset_index)
            .join(QualitySignal, QualitySignal.example_id == Example.id)
            .where(QualitySignal.flagged == True)  # noqa: E712
            .distinct()
        ).scalars().all()
    )
    session.close()

    flagged_indices.sort()
    out_path = "flagged_indices.json"
    with open(out_path, "w") as f:
        json.dump(flagged_indices, f)

    print(f"Exported {len(flagged_indices)} flagged indices to {out_path}")


if __name__ == "__main__":
    main()
