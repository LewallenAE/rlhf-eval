"""Evaluation utilities for reward models."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .model import RewardModel


def evaluate_reward_model(
    model: RewardModel,
    dataset,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict:
    """Evaluate reward model accuracy on a preference pair dataset.

    Returns:
        Dict with accuracy, correct, total, and avg_reward_gap.
    """
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct, total = 0, 0
    total_gap = 0.0

    with torch.no_grad():
        try:
            from tqdm import tqdm
            iterator = tqdm(loader, desc="Evaluating")
        except ImportError:
            iterator = loader

        for batch in iterator:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            reward_chosen = model(chosen_ids, chosen_mask)
            reward_rejected = model(rejected_ids, rejected_mask)

            gap = reward_chosen - reward_rejected
            correct += (gap > 0).sum().item()
            total_gap += gap.sum().item()
            total += chosen_ids.size(0)

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "avg_reward_gap": total_gap / total if total > 0 else 0.0,
    }
