"""DeBERTa-based prompt-router classifier (stage-1 replacement).

Replaces GBNF-constrained JSON decoding on the local GGUF LLM with a
dedicated ~70M-parameter transformer classifier fine-tuned for this router:
    hub repo:  xubayer/prompt-router-deberta-v3-xsmall
    baked to:  /models/router  (see scripts/bake_router_model.py + Dockerfile)

The classifier is a shared DeBERTa-v3-xsmall backbone with mean pooling and
two linear heads (intent: 8-way, difficulty: shallow/deep). `confidence` is
the softmax probability of the predicted intent — a float in [0,1] that
feeds the existing `local_confidence_threshold` gate directly.

This module deliberately imports torch/transformers lazily and only inside
RouterClassifier so that offline dev machines and CI without those packages
still run everything else (they fall back to the llama.cpp / heuristic
classify paths in loader.py).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

DEFAULT_ROUTER_PATH = "/models/router"

# Canonical label order used by the training script. Overridden by
# labels.json in the model snapshot when present, so a retrained model with
# a different order still resolves correctly.
INTENTS = [
    "factual_knowledge",
    "math_reasoning",
    "sentiment",
    "summarization",
    "ner",
    "code_debugging",
    "logical_reasoning",
    "code_generation",
]
DIFFICULTIES = ["shallow", "deep"]

# Weight files tried in order when loading the fine-tuned state dict.
_WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin", "best_model.pt"]


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    if all(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


class RouterClassifier:
    """Loads the baked snapshot fully OFFLINE and classifies prompts.

    Raises on any load problem — the caller (loader.LocalModel) treats a
    raised constructor as "classifier unavailable" and falls back to the
    older classify paths, so a broken snapshot degrades instead of crashing
    the container.
    """

    def __init__(self, model_dir: Optional[str] = None, max_length: int = 256):
        model_dir = Path(model_dir or os.environ.get("ROUTER_MODEL_PATH", DEFAULT_ROUTER_PATH))
        if not model_dir.is_dir():
            raise FileNotFoundError(f"router model dir not found: {model_dir}")

        import torch
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self._torch = torch
        self.max_length = max_length

        labels_path = model_dir / "labels.json"
        if labels_path.is_file():
            labels = json.load(open(labels_path, encoding="utf-8"))
            self.intents = labels.get("intents", INTENTS)
            self.difficulties = labels.get("difficulties", DIFFICULTIES)
        else:
            self.intents = INTENTS
            self.difficulties = DIFFICULTIES

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # The backbone is rebuilt from a LOCAL config (baked at build time by
        # scripts/bake_router_model.py) so no network is ever touched at
        # runtime; its weights come from the fine-tuned state dict below.
        backbone_cfg_dir = model_dir / "backbone_config"
        cfg_src = str(backbone_cfg_dir if backbone_cfg_dir.is_dir() else model_dir)
        backbone_cfg = AutoConfig.from_pretrained(cfg_src)
        backbone = AutoModel.from_config(backbone_cfg)
        hidden = backbone.config.hidden_size

        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.intent_head = nn.Linear(hidden, len(INTENTS))
                self.difficulty_head = nn.Linear(hidden, len(DIFFICULTIES))

            def forward(self, input_ids, attention_mask):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                return self.intent_head(pooled), self.difficulty_head(pooled)

        self.model = _Module()
        self._load_weights(model_dir)
        self.model.eval()
        torch.set_num_threads(int(os.environ.get("LOCAL_MODEL_N_THREADS", "2")))
        self.last_latency_ms: float = 0.0

    def _load_weights(self, model_dir: Path) -> None:
        torch = self._torch
        state_dict = None
        for name in _WEIGHT_FILES:
            path = model_dir / name
            if not path.is_file():
                continue
            if name.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(str(path))
            else:
                state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
            break
        if state_dict is None:
            raise FileNotFoundError(
                f"no weight file found in {model_dir} (tried {_WEIGHT_FILES})"
            )
        for prefix in ("module.", "model."):
            state_dict = _strip_prefix(state_dict, prefix)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        # The heads MUST load — silently random heads would classify garbage.
        head_keys = [k for k in missing if "head" in k]
        if head_keys:
            raise RuntimeError(
                f"classifier head weights missing from snapshot: {head_keys}; "
                f"unexpected keys sample: {list(unexpected)[:8]}"
            )
        # Backbone key mismatches beyond a handful likewise mean the snapshot
        # doesn't match this architecture — better to fall back than mis-route.
        backbone_missing = [k for k in missing if k.startswith("backbone.")]
        if len(backbone_missing) > 4:
            raise RuntimeError(
                f"{len(backbone_missing)} backbone weights missing from snapshot "
                f"(sample: {backbone_missing[:5]}) — snapshot/architecture mismatch"
            )

    def classify(self, prompt: str, max_length: Optional[int] = None) -> dict:
        """Return {"intent","difficulty","confidence"} — drop-in for the
        LLM-based classify. confidence is a float in [0,1]."""
        torch = self._torch
        start = time.time()
        enc = self.tokenizer(
            prompt, truncation=True, max_length=max_length or self.max_length,
            padding="max_length", return_tensors="pt",
        )
        with torch.no_grad():
            intent_logits, difficulty_logits = self.model(
                enc["input_ids"], enc["attention_mask"]
            )
        intent_probs = torch.softmax(intent_logits, dim=-1)[0]
        ip = int(intent_probs.argmax())
        dp = int(difficulty_logits[0].argmax())
        self.last_latency_ms = (time.time() - start) * 1000
        return {
            "intent": self.intents[ip],
            "difficulty": self.difficulties[dp],
            "confidence": round(float(intent_probs[ip]), 3),
        }
