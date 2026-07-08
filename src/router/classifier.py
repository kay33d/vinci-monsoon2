"""Stage 1 — local intent/difficulty classification (0 scored tokens)."""

from __future__ import annotations

from src.local_models.loader import LocalModel


def classify_task(task_prompt: str, model: LocalModel, max_chars: int = 1500,
                  max_tokens: int = 64) -> dict:
    """Classify one task prompt into {"intent","difficulty","confidence"}.

    Long prompts (e.g. big summarization passages) are truncated for the
    classifier only — the intent is obvious from the opening instruction, and
    a short stage-1 prompt keeps CPU latency well inside the 30s budget. The
    FULL prompt is always used for the actual answer in stage 2.
    """
    snippet = task_prompt if len(task_prompt) <= max_chars else task_prompt[:max_chars] + " ..."
    return model.classify(snippet, max_tokens=max_tokens)
