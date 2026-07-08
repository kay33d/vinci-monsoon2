"""Stage 2 — routing decision + category->role->model resolution.

Decision flow per task:
  1. classify locally (0 tokens)
  2. apply the per-category escalation policy from config/routing_map.yaml
  3. answer locally (0 tokens) OR escalate to the role-resolved Fireworks model
  4. if the remote call fails, fall back to local generate() — never emit an
     empty answer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from config.prompts import REMOTE_SYSTEM, remote_user_prompt
from src.api_clients.fireworks import FireworksClient, FireworksError
from src.local_models.loader import LocalModel
from src.router.classifier import classify_task

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "routing_map.yaml"


def load_config(path: Optional[Path] = None) -> dict:
    with open(path or _CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def allowed_models() -> list[str]:
    """Parse ALLOWED_MODELS from the environment (never hardcoded)."""
    raw = os.environ.get("ALLOWED_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


class Router:
    def __init__(self, local_model: LocalModel, fireworks: FireworksClient,
                 config: Optional[dict] = None):
        self.local = local_model
        self.fireworks = fireworks
        self.cfg = config or load_config()
        self.allowed = allowed_models()
        self.limits = self.cfg.get("limits", {})

    # ------------------------------------------------------------------ #
    # category -> role -> concrete allowed model ID                       #
    # ------------------------------------------------------------------ #
    def resolve_model(self, category: str) -> Optional[str]:
        """Map a category to a model ID present in ALLOWED_MODELS.

        The hint table is best-effort: any hint substring that matches an
        allowed ID (case-insensitive) wins; otherwise degrade to the first
        allowed model. Returns None only when ALLOWED_MODELS is empty, in
        which case the caller must answer locally.
        """
        if not self.allowed:
            return None
        role = self.cfg["category_roles"].get(category, "general")
        for hint in self.cfg["role_model_hints"].get(role, []):
            for model_id in self.allowed:
                if hint.lower() in model_id.lower():
                    return model_id
        return self.allowed[0]

    # ------------------------------------------------------------------ #
    # escalation decision                                                #
    # ------------------------------------------------------------------ #
    def should_answer_locally(self, decision: dict) -> bool:
        policy_cfg = self.cfg.get("escalation_policy", {})
        policy = policy_cfg.get("overrides", {}).get(
            decision["intent"], policy_cfg.get("default", "strict")
        )
        if policy == "always":
            return False
        if policy == "lenient":
            return decision["difficulty"] == "shallow"
        # strict (default): both conditions must hold
        return decision["difficulty"] == "shallow" and decision["confidence"] == "high"

    # ------------------------------------------------------------------ #
    # full pipeline for one task                                         #
    # ------------------------------------------------------------------ #
    def route(self, task_prompt: str) -> tuple[str, dict]:
        """Return (answer, meta). meta records how the task was routed."""
        decision = classify_task(
            task_prompt, self.local,
            max_chars=self.limits.get("classify_prompt_chars", 1500),
            max_tokens=self.limits.get("classify_max_tokens", 64),
        )
        category = decision["intent"]
        meta = {"decision": decision, "route": "local", "model": "local"}

        go_local = self.should_answer_locally(decision)
        model_id = None if go_local else self.resolve_model(category)
        if model_id is None:
            # Either policy says local, or ALLOWED_MODELS is empty.
            answer = self._local_answer(task_prompt)
            return answer, meta

        try:
            answer = self.fireworks.chat(
                model=model_id,
                system=REMOTE_SYSTEM,
                user=remote_user_prompt(category, task_prompt),
                max_tokens=self.limits.get("remote_max_tokens", 512),
                timeout=self.limits.get("remote_timeout_seconds", 25),
            )
            meta.update(route="remote", model=model_id)
            return answer, meta
        except FireworksError as exc:
            # Remote failed after its retry — degrade to local, never crash.
            meta.update(route="local_fallback", model="local", error=str(exc))
            return self._local_answer(task_prompt), meta

    def _local_answer(self, task_prompt: str) -> str:
        return self.local.generate(
            task_prompt, max_tokens=self.limits.get("local_max_tokens", 300)
        )
