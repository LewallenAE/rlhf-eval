"""Training loop for the reward model using Bradley-Terry pairwise loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import RewardModel


def train_reward_model(
    model: RewardModel,
    dataset,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 2e-5,
    device: str = "cuda",
) -> dict:
    """Train a reward model with Bradley-Terry pairwise loss.

    Returns:
        Dict with per-epoch loss and accuracy, plus final metrics.
    """
    model = model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"epoch_loss": [], "epoch_accuracy": []}

    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0

        try:
            from tqdm import tqdm
            iterator = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        except ImportError:
            iterator = loader

        for batch in iterator:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            reward_chosen = model(chosen_ids, chosen_mask)
            reward_rejected = model(rejected_ids, rejected_mask)

            # Bradley-Terry: -log sigmoid(r_chosen - r_rejected)
            loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * chosen_ids.size(0)
            correct += (reward_chosen > reward_rejected).sum().item()
            total += chosen_ids.size(0)

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        history["epoch_loss"].append(epoch_loss)
        history["epoch_accuracy"].append(epoch_acc)
        print(f"  Epoch {epoch + 1}: loss={epoch_loss:.4f}, accuracy={epoch_acc:.4f}")

    return {
        "final_loss": history["epoch_loss"][-1],
        "final_accuracy": history["epoch_accuracy"][-1],
        "history": history,
    }
