"""Reward model wrapping DistilBERT for scalar reward prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class RewardModel(nn.Module):
    """Bradley-Terry reward model: backbone -> [CLS] -> Linear -> scalar."""

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar reward for each example in the batch."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        reward = self.reward_head(cls_embedding).squeeze(-1)  # (batch,)
        return reward
