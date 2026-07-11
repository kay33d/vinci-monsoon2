"""Stage 2 — routing decision + category->role->model resolution.

Decision flow per task:
  1. classify locally (0 tokens)
  2. apply the per-category escalation policy (config/routing_map.yaml)
  3. apply the AGGRESSIVE override: prompts containing code syntax, math, or
     step-by-step cues escalate even when the classifier calls them easy
  4. answer locally (0 tokens) OR escalate to the role-resolved Fireworks
     model; if the primary remote model fails, try the OTHER allowed model
     before falling back to local — never emit an empty answer.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import yaml

from config.prompts import REMOTE_SYSTEM, remote_user_prompt
from src.api_clients.fireworks import FireworksClient, FireworksError
from src.local_models.loader import LocalModel
from src.router.classifier import classify_task

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "routing_map.yaml"

# The LLM/heuristic classify paths emit categorical confidence; map it to a
# score so the config threshold (local_confidence_threshold) can gate it
# numerically. The DeBERTa router path emits a float directly (softmax prob
# of the predicted intent), which feeds the same gate untranslated.
_CONFIDENCE_SCORE = {"high": 0.95, "low": 0.50}


def _confidence_value(raw) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    return _CONFIDENCE_SCORE.get(raw, 0.0)

# --- Aggressive escalation cues (checked BEFORE accepting a local answer) ---
# Code syntax / language names
_CODE_CUES = re.compile(
    r"(def |class |return\b|function\b|=>|;|\{|\}|import |print\(|"
    r"\bpython\b|\bjavascript\b|\bjava\b|\bc\+\+\b|\bsql\b|\brust\b|\bbug\b)",
    re.IGNORECASE,
)
# Math symbols / multi-step calculation words
_MATH_CUES = re.compile(
    r"(\d+\s*[+\-*/^=]\s*\d+|%|\bpercent|\bcalculate\b|\bhow many\b|\baverage\b|"
    r"\bsum\b|\btotal\b|\bprofit\b|\bratio\b|\brate\b|\bprojection\b)",
    re.IGNORECASE,
)
# Explicit reasoning cues
_REASONING_CUES = re.compile(
    r"(step[- ]by[- ]step|explain your reasoning|deduce|puzzle|constraint|"
    r"each own|who owns|what is the order)",
    re.IGNORECASE,
)


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
        self.thresholds = self.cfg.get("thresholds", {})

    # ------------------------------------------------------------------ #
    # role -> concrete allowed model ID                                   #
    # ------------------------------------------------------------------ #
    def resolve_role(self, role: str) -> Optional[str]:
        """Resolve a role to a model ID present in ALLOWED_MODELS.

        Hint substrings are matched case-insensitively; graceful fallback to
        the first allowed model; None only if ALLOWED_MODELS is empty.
        """
        if not self.allowed:
            return None
        for hint in self.cfg["role_model_hints"].get(role, []):
            for model_id in self.allowed:
                if hint.lower() in model_id.lower():
                    return model_id
        return self.allowed[0]

    def resolve_model(self, category: str) -> Optional[str]:
        return self.resolve_role(self.cfg["category_roles"].get(category, "general"))

    def resolved_map(self) -> dict:
        """role -> model ID map, for the startup log."""
        roles = sorted(set(self.cfg["category_roles"].values()))
        return {role: self.resolve_role(role) for role in roles}

    # ------------------------------------------------------------------ #
    # escalation decision                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def aggressive_override(prompt: str) -> Optional[str]:
        """Return the cue type if the prompt should escalate regardless of
        the classifier's opinion, else None."""
        if _CODE_CUES.search(prompt):
            return "code_cue"
        if _MATH_CUES.search(prompt):
            return "math_cue"
        if _REASONING_CUES.search(prompt):
            return "reasoning_cue"
        return None

    def should_answer_locally(self, decision: dict, prompt: str) -> bool:
        policy_cfg = self.cfg.get("escalation_policy", {})
        policy = policy_cfg.get("overrides", {}).get(
            decision["intent"], policy_cfg.get("default", "strict")
        )
        if policy == "always":
            return False
        if decision["difficulty"] != "shallow":
            return False
        # Confidence gate: the score (float from the router, or mapped from
        # the categorical LLM/heuristic labels) must EXCEED the config
        # threshold (default 0.90) to stay local.
        conf = _confidence_value(decision.get("confidence"))
        if conf <= self.thresholds.get("local_confidence_threshold", 0.90):
            return False
        # Aggressive cue check runs LAST, before accepting a local answer.
        if self.thresholds.get("aggressive_escalation", True) and \
                self.aggressive_override(prompt):
            return False
        return True

    # ------------------------------------------------------------------ #
    # full pipeline for one task                                         #
    # ------------------------------------------------------------------ #
    def route(self, task_prompt: str) -> tuple[str, dict]:
        """Return (answer, meta). meta records how the task was routed."""
        classify_start = time.time()
        decision = classify_task(
            task_prompt, self.local,
            max_chars=self.limits.get("classify_prompt_chars", 1500),
            max_tokens=self.limits.get("classify_max_tokens", 64),
        )
        category = decision["intent"]
        meta = {
            "decision": decision, "route": "local", "model": "local",
            "finish_reason": "-", "truncated": False,
            "escalation_cue": self.aggressive_override(task_prompt) or "-",
            "classify_ms": round((time.time() - classify_start) * 1000, 1),
        }

        if self.should_answer_locally(decision, task_prompt):
            return self._local_answer(task_prompt), meta

        primary = self.resolve_model(category)
        if primary is None:  # ALLOWED_MODELS empty: local is all we have
            return self._local_answer(task_prompt), meta

        role = self.cfg["category_roles"].get(category, "general")
        max_tokens = self.limits.get("remote_max_tokens_by_role", {}).get(
            role, self.limits.get("remote_max_tokens", 512)
        )
        # Primary model, then the first DIFFERENT allowed model, then local.
        alternate = next((m for m in self.allowed if m != primary), None)
        last_err: Optional[Exception] = None
        for model_id in [m for m in (primary, alternate) if m]:
            try:
                answer = self.fireworks.chat(
                    model=model_id,
                    system=REMOTE_SYSTEM,
                    user=remote_user_prompt(category, task_prompt),
                    max_tokens=max_tokens,
                    timeout=self.limits.get("remote_timeout_seconds", 25),
                )
                finish = getattr(self.fireworks, "last_finish_reason", None)
                meta.update(
                    route="remote", model=model_id,
                    finish_reason=finish or "?",
                    truncated=finish == "length",
                )
                return answer, meta
            except FireworksError as exc:
                last_err = exc

        # Both remote attempts failed — degrade to local, never crash.
        meta.update(route="local_fallback", model="local", error=str(last_err))
        return self._local_answer(task_prompt), meta

    def _local_answer(self, task_prompt: str) -> str:
        return self.local.generate(
            task_prompt, max_tokens=self.limits.get("local_max_tokens", 300)
        )
