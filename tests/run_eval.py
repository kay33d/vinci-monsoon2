"""Offline eval harness — runs the FULL router over mock_tasks.json with
Fireworks MOCKED, so it needs no API key, no network, and no GGUF model.

Usage:  python tests/run_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# A plausible-looking allowed list purely for exercising role resolution
# offline. The real list comes from the harness at evaluation time.
os.environ.setdefault(
    "ALLOWED_MODELS",
    "accounts/fireworks/models/gemma-3-27b-it,"
    "accounts/fireworks/models/minimax-m2,"
    "accounts/fireworks/models/kimi-k2-instruct",
)

from src.local_models.loader import get_local_model      # noqa: E402
from src.router.dispatch import Router                   # noqa: E402

MOCK_TASKS = ROOT / "tests" / "mock_tasks.json"


class MockFireworks:
    """Stub Fireworks client: canned answer + simulated token accounting."""

    def __init__(self):
        self.total_tokens = 0
        self.calls = 0

    @property
    def configured(self) -> bool:
        return True

    def chat(self, model, system, user, max_tokens=512, timeout=25.0) -> str:
        # Rough token simulation: ~1 token per 4 chars of input + 60 output.
        self.total_tokens += (len(system) + len(user)) // 4 + 60
        self.calls += 1
        return f"[mock answer from {model.split('/')[-1]}]"


def run_offline_eval(output_path: Path | None = None) -> list[dict]:
    with open(MOCK_TASKS, "r", encoding="utf-8") as fh:
        tasks = json.load(fh)

    mock = MockFireworks()
    router = Router(get_local_model(), mock)

    results, rows = [], defaultdict(lambda: {"local": 0, "remote": 0, "model": "-"})
    for task in tasks:
        answer, meta = router.route(task["prompt"])
        results.append({"task_id": task["task_id"], "answer": answer})
        cat = meta["decision"]["intent"]
        if meta["route"] == "remote":
            rows[cat]["remote"] += 1
            rows[cat]["model"] = meta["model"].split("/")[-1]
        else:
            rows[cat]["local"] += 1

    # --- report ---------------------------------------------------------
    print(f"\nlocal backend: {router.local.backend}")
    print(f"{'category':<20} {'local':>5} {'remote':>6}  remote model")
    print("-" * 60)
    for cat in sorted(rows):
        r = rows[cat]
        print(f"{cat:<20} {r['local']:>5} {r['remote']:>6}  {r['model']}")
    print("-" * 60)
    print(f"tasks: {len(results)} | fireworks calls: {mock.calls} | "
          f"simulated tokens: {mock.total_tokens}\n")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=1)
    return results


if __name__ == "__main__":
    run_offline_eval(ROOT / "tests" / "out_results.json")
