"""CPU-only GGUF model loader with grammar-constrained classification.

Backends, in order of preference:
  1. llama-cpp-python + a local GGUF (the real path used inside the container).
     Stage-1 classification uses a GBNF grammar so the output is ALWAYS the
     exact {"intent","difficulty","confidence"} JSON — we never best-effort
     parse free text.
  2. A deterministic keyword heuristic. Used only when llama-cpp-python or the
     model file is missing (e.g. offline dev machines / CI). It keeps the whole
     pipeline runnable and testable with zero downloads.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Optional

from config.prompts import CLASSIFY_SYSTEM, INTENTS, LOCAL_ANSWER_SYSTEM

DEFAULT_MODEL_PATH = "/models/model.gguf"

# GBNF grammar that forces the exact classification JSON shape. With this,
# llama.cpp cannot emit anything except a valid decision object.
_CLASSIFY_GRAMMAR = r"""
root ::= "{" ws "\"intent\"" ws ":" ws intent ws "," ws "\"difficulty\"" ws ":" ws difficulty ws "," ws "\"confidence\"" ws ":" ws confidence ws "}"
intent ::= "\"factual_knowledge\"" | "\"math_reasoning\"" | "\"sentiment\"" | "\"summarization\"" | "\"ner\"" | "\"code_debugging\"" | "\"logical_reasoning\"" | "\"code_generation\""
difficulty ::= "\"shallow\"" | "\"deep\""
confidence ::= "\"high\"" | "\"low\""
ws ::= [ \t\n]*
"""

_DEFAULT_DECISION = {"intent": "factual_knowledge", "difficulty": "deep", "confidence": "low"}


class LocalModel:
    """Wraps either a llama.cpp model or the heuristic fallback."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.environ.get("LOCAL_MODEL_PATH", DEFAULT_MODEL_PATH)
        self._llm = None
        self._grammar = None
        self.backend = "heuristic"
        self.classifier_backend = "heuristic"
        # llama-cpp-python's Llama object is not safe to call concurrently
        # from multiple threads on the same instance — the entrypoint's
        # per-task timeout abandons slow calls (leaves their thread running)
        # rather than killing them, so a later task can otherwise end up
        # calling into this same model while an abandoned call is still
        # in flight. That concurrent access segfaults the process. Every
        # call into self._llm is serialized through this lock instead.
        self._lock = threading.Lock()
        self._router = self._try_load_router()
        self._try_load_llama()
        if self._router is not None:
            self.classifier_backend = "deberta-router"
        elif self._llm is not None:
            self.classifier_backend = "llama.cpp"

    @staticmethod
    def _try_load_router():
        """Stage-1 classifier: the baked DeBERTa router (see
        router_classifier.py). Any load failure degrades to the older
        classify paths — a running agent with weaker routing beats a
        crashed container."""
        try:
            from src.local_models.router_classifier import RouterClassifier
            return RouterClassifier()
        except Exception as exc:
            print(f"router classifier unavailable ({exc}); "
                  "falling back to llama.cpp/heuristic classify", flush=True)
            return None

    def _try_load_llama(self) -> None:
        if not os.path.isfile(self.model_path):
            return
        try:
            from llama_cpp import Llama, LlamaGrammar

            # Defaults are deliberately conservative: the grading VM is
            # 4GB RAM / 2 vCPU. A cgroup OOM kill is SIGKILL — it cannot be
            # caught by the except below, so the only real defense is
            # keeping actual memory use (weights + KV cache + compute
            # buffers) well under the cap rather than relying on graceful
            # fallback after the fact.
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=int(os.environ.get("LOCAL_MODEL_N_CTX", "2048")),
                n_threads=int(os.environ.get("LOCAL_MODEL_N_THREADS", "2")),
                n_batch=int(os.environ.get("LOCAL_MODEL_N_BATCH", "256")),
                n_gpu_layers=0,  # grading VM is CPU-only
                verbose=False,
            )
            self._grammar = LlamaGrammar.from_string(_CLASSIFY_GRAMMAR)
            self.backend = "llama.cpp"
        except Exception:
            # Any load failure (bad file, missing lib) degrades to heuristic —
            # a running agent with weaker routing beats a crashed container.
            self._llm = None

    # ------------------------------------------------------------------ #
    # Stage 1: classification                                            #
    # ------------------------------------------------------------------ #
    def classify(self, task_prompt: str, max_tokens: int = 64) -> dict:
        """Return {"intent","difficulty","confidence"} — always valid.

        Preference order: DeBERTa router (fast, dedicated classifier) ->
        llama.cpp GBNF decoding -> keyword heuristic. The router returns a
        FLOAT confidence (softmax prob of the predicted intent); the older
        paths return categorical "high"/"low" — dispatch handles both.
        """
        if self._router is not None:
            try:
                return self._validate(self._router.classify(task_prompt))
            except Exception:
                pass
        if self._llm is not None:
            try:
                with self._lock:
                    out = self._llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": CLASSIFY_SYSTEM},
                            {"role": "user", "content": task_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.0,
                        grammar=self._grammar,
                    )
                decision = json.loads(out["choices"][0]["message"]["content"])
                return self._validate(decision)
            except Exception:
                pass
        return self._heuristic_classify(task_prompt)

    @staticmethod
    def _validate(decision: dict) -> dict:
        """Belt-and-braces: never trust IO. Confidence may be categorical
        ("high"/"low", from the LLM/heuristic paths) or a float in [0,1]
        (from the DeBERTa router)."""
        conf = decision.get("confidence")
        conf_ok = conf in ("high", "low") or (
            isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0
        )
        if (
            decision.get("intent") in INTENTS
            and decision.get("difficulty") in ("shallow", "deep")
            and conf_ok
        ):
            return decision
        return dict(_DEFAULT_DECISION)

    # ------------------------------------------------------------------ #
    # Stage 2a: local generation                                         #
    # ------------------------------------------------------------------ #
    def generate(self, task_prompt: str, max_tokens: int = 300) -> str:
        if self._llm is not None:
            try:
                with self._lock:
                    out = self._llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": LOCAL_ANSWER_SYSTEM},
                            {"role": "user", "content": task_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.2,
                    )
                text = out["choices"][0]["message"]["content"].strip()
                if text:
                    return text
            except Exception:
                pass
        return self._heuristic_generate(task_prompt)

    # ------------------------------------------------------------------ #
    # Heuristic fallback backend (dev machines / emergency only)          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _heuristic_classify(prompt: str) -> dict:
        p = prompt.lower()
        if "sentiment" in p or "classify the sentiment" in p:
            return {"intent": "sentiment", "difficulty": "shallow", "confidence": "high"}
        if "summar" in p:
            return {"intent": "summarization", "difficulty": "shallow", "confidence": "high"}
        if "entit" in p or "named entity" in p:
            return {"intent": "ner", "difficulty": "shallow", "confidence": "high"}
        if ("bug" in p or "fix" in p) and ("def " in p or "function" in p or "code" in p):
            return {"intent": "code_debugging", "difficulty": "deep", "confidence": "high"}
        if re.search(r"\bwrite\b.*\b(function|code|program|script)\b", p):
            return {"intent": "code_generation", "difficulty": "deep", "confidence": "high"}
        if re.search(r"\d", p) and re.search(r"\b(how many|calculate|percent|%|remain|total|sum|cost)\b", p):
            return {"intent": "math_reasoning", "difficulty": "deep", "confidence": "high"}
        if re.search(r"\b(each own|who owns|puzzle|constraint|deduce|logic)\b", p):
            return {"intent": "logical_reasoning", "difficulty": "deep", "confidence": "high"}
        return {"intent": "factual_knowledge", "difficulty": "shallow", "confidence": "low"}

    @staticmethod
    def _heuristic_generate(prompt: str) -> str:
        # Last-resort non-empty answer; only reachable when no GGUF is loaded.
        head = re.sub(r"\s+", " ", prompt).strip()[:160]
        return f"Best-effort response (local model unavailable) to: {head}"


_singleton: Optional[LocalModel] = None


def get_local_model() -> LocalModel:
    global _singleton
    if _singleton is None:
        _singleton = LocalModel()
    return _singleton
