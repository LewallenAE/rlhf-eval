    #!/usr/bin/env python3
"""
Conversation parsing utilities for HH-RLHF dataset.

HH-RLHF conversations follow the format:
    "Human: text\\n\\nAssistant: text\\n\\nHuman: text\\n\\nAssistant: text"

This module extracts structured turns from that raw string format.
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
import re

# ----------------- Third Party Library -----------------

# ----------------- Application Imports -----------------

# ----------------- Module-level Configuration -----------------

SPEAKER_RE = re.compile(r"^(Human|Assistant):\s*", re.MULTILINE)


def parse_conversation(text: str) -> list[tuple[str, str]]:
    """
    Parse a raw conversation string into a list of (role, content) tuples.

    Args:
        text: Raw conversation string with "Human:" and "Assistant:" prefixes.

    Returns:
        List of tuples: [("human", "content"), ("assistant", "content"), ...].
        Returns an empty list if *text* is empty or contains no speaker tags.
    """
    if not text or not text.strip():
        return []

    parts = SPEAKER_RE.split(text)
    # parts layout: [preamble, role, content, role, content, ...]
    # Index 0 is anything before the first speaker tag (usually empty).
    # Roles at odd indices, content at even indices starting from 2.

    turns: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        role = parts[i].lower()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        turns.append((role, content))

    return turns


def last_assistant_response(text: str) -> str:
    """
    Extract the final assistant turn from a conversation.

    Args:
        text: Raw conversation string.

    Returns:
        The content of the last assistant turn, or empty string if none found.
    """
    turns = parse_conversation(text)
    for role, content in reversed(turns):
        if role == "assistant":
            return content
    return ""


def last_human_prompt(text: str) -> str:
    """
    Extract the final human turn from a conversation.

    Args:
        text: Raw conversation string.

    Returns:
        The content of the last human turn, or empty string if none found.
    """
    turns = parse_conversation(text)
    for role, content in reversed(turns):
        if role == "human":
            return content
    return ""


def all_assistant_responses(text: str) -> str:
    """
    Concatenate all assistant turns separated by newlines.

    Args:
        text: Raw conversation string.

    Returns:
        All assistant turn contents joined by ``"\\n"``.
    """
    turns = parse_conversation(text)
    return "\n".join(content for role, content in turns if role == "assistant")


def all_human_prompts(text: str) -> str:
    """
    Concatenate all human turns separated by newlines.

    Args:
        text: Raw conversation string.

    Returns:
        All human turn contents joined by ``"\\n"``.
    """
    turns = parse_conversation(text)
    return "\n".join(content for role, content in turns if role == "human")


def count_turns(text: str) -> dict[str, int]:
    """
    Count the number of turns by role.

    Args:
        text: Raw conversation string.

    Returns:
        Dictionary with keys ``"human"`` and ``"assistant"`` mapped to counts.
    """
    turns = parse_conversation(text)
    counts: dict[str, int] = {"human": 0, "assistant": 0}
    for role, _ in turns:
        counts[role] = counts.get(role, 0) + 1
    return counts
