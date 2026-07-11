"""BUILD-TIME script: bake the DeBERTa router classifier into the image.

Runs inside the Docker builder stage (network available). Never runs on the
grading VM. It:
  1. snapshot_downloads the classifier repo to --out (default /models/router)
  2. saves the backbone's config locally so the runtime can rebuild the
     architecture with ZERO network access
  3. loads the fine-tuned weights through the exact runtime code path
     (src/local_models/router_classifier.py) and runs an 8-category smoke
     evaluation — the build FAILS below the 68% routing-accuracy target,
     so a bad snapshot can never ship
  4. writes /models/router/BAKE_DIAG.txt recording the validation results

Usage (see Dockerfile):
    python scripts/bake_router_model.py \
        --model-id xubayer/prompt-router-deberta-v3-xsmall --out /models/router
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

FALLBACK_BACKBONE = "microsoft/deberta-v3-xsmall"

# The 8 primary capability mapping prompts + expected intents. 68% of 8
# rounds up to 6 — at least 6 must route correctly for the build to pass.
SMOKE_PROMPTS = [
    ("Explain how a refrigerator keeps food cold.", "factual_knowledge"),
    ("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?", "math_reasoning"),
    ("Classify the sentiment of this review: The battery life is great, but the screen scratches too easily.", "sentiment"),
    ("Summarize the following in exactly one sentence: The new library opened downtown last month and offers free coding classes.", "summarization"),
    ("Extract all named entities and their types from: Maria Sanchez joined Fireworks AI in Berlin last March.", "ner"),
    ("This function should return the max of a list but has a bug: def get_max(nums): return nums[0]. Find and fix it.", "code_debugging"),
    ("Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?", "logical_reasoning"),
    ("Write a Python function that returns the second-largest number in a list, handling duplicates correctly.", "code_generation"),
]
MIN_CORRECT = 6  # ceil(0.68 * 8)


def find_backbone_name(snapshot_dir: Path) -> str:
    """Best-effort recovery of the backbone model name from the snapshot's
    config.json; falls back to the known training backbone."""
    cfg_path = snapshot_dir / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.load(open(cfg_path, encoding="utf-8"))

            def walk(obj):
                if isinstance(obj, str) and "deberta" in obj.lower():
                    return obj
                if isinstance(obj, dict):
                    for v in obj.values():
                        hit = walk(v)
                        if hit:
                            return hit
                if isinstance(obj, list):
                    for v in obj:
                        hit = walk(v)
                        if hit:
                            return hit
                return None

            hit = walk(cfg)
            if hit and "/" in hit:
                return hit
        except Exception as exc:
            print(f"bake: could not parse snapshot config.json ({exc}); using fallback backbone")
    return FALLBACK_BACKBONE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", default="/models/router")
    args = ap.parse_args()

    out = Path(args.out)
    diag_lines: list[str] = []

    def diag(line: str):
        print(line, flush=True)
        diag_lines.append(line)

    # ---- 1. download the classifier snapshot -------------------------------
    from huggingface_hub import snapshot_download
    snapshot_download(args.model_id, local_dir=str(out))
    diag(f"BAKE | snapshot | {args.model_id} -> {out}")
    diag(f"BAKE | snapshot files | {sorted(p.name for p in out.iterdir())}")

    # ---- 2. make the snapshot fully offline-loadable ------------------------
    from transformers import AutoConfig, AutoTokenizer

    backbone_name = find_backbone_name(out)
    backbone_cfg = AutoConfig.from_pretrained(backbone_name)
    (out / "backbone_config").mkdir(exist_ok=True)
    backbone_cfg.save_pretrained(str(out / "backbone_config"))
    diag(f"BAKE | backbone config | {backbone_name} -> {out}/backbone_config")

    try:
        AutoTokenizer.from_pretrained(str(out))
        diag("BAKE | tokenizer | present in snapshot")
    except Exception:
        AutoTokenizer.from_pretrained(backbone_name).save_pretrained(str(out))
        diag(f"BAKE | tokenizer | missing from snapshot, saved from {backbone_name}")

    # ---- 3. validate through the EXACT runtime load path -------------------
    from router_classifier import RouterClassifier  # PYTHONPATH set by Dockerfile

    clf = RouterClassifier(model_dir=str(out))
    diag("BAKE | weights | loaded through runtime RouterClassifier (schema mapping OK)")

    correct = 0
    latencies = []
    for prompt, expected in SMOKE_PROMPTS:
        decision = clf.classify(prompt)
        latencies.append(clf.last_latency_ms)
        ok = decision["intent"] == expected
        correct += ok
        diag(
            f"BAKE-DIAG | {expected:<18} | got={decision['intent']:<18} | "
            f"diff={decision['difficulty']:<7} | conf={decision['confidence']:.3f} | "
            f"{clf.last_latency_ms:.0f}ms | {'OK' if ok else 'MISS'}"
        )
        # Contract check: exact evaluation-contract keys and types.
        assert set(decision) == {"intent", "difficulty", "confidence"}, decision
        assert isinstance(decision["confidence"], float)

    acc = correct / len(SMOKE_PROMPTS)
    diag(
        f"BAKE | smoke accuracy | {correct}/{len(SMOKE_PROMPTS)} = {acc:.0%} "
        f"(target >= 68%) | mean latency {sum(latencies)/len(latencies):.0f}ms | "
        f"max latency {max(latencies):.0f}ms"
    )

    (out / "BAKE_DIAG.txt").write_text("\n".join(diag_lines) + "\n", encoding="utf-8")

    if correct < MIN_CORRECT:
        print(
            f"BAKE FAILED: routing accuracy {acc:.0%} below the 68% target — "
            "refusing to ship this classifier snapshot",
            file=sys.stderr,
        )
        return 1
    print("BAKE OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
