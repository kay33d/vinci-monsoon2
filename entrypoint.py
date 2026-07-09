"""Container entrypoint for the Track 1 hybrid routing agent.

Contract with the grading harness:
  * read  /input/tasks.json   (list of {"task_id", "prompt"})
  * write /output/results.json (list of {"task_id", "answer"}) before exiting
  * exit 0 on success, non-zero on failure
  * total runtime < 10 min, per-request < 30 s, ready < 60 s

INPUT_PATH / OUTPUT_PATH env vars exist only for local development; the
defaults match the harness contract.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Make repo-root imports work no matter where the script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.api_clients.fireworks import FireworksClient          # noqa: E402
from src.local_models.loader import get_local_model            # noqa: E402
from src.router.dispatch import Router                         # noqa: E402

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# Soft ceiling on total runtime, configurable via env. Once elapsed time
# crosses it we STOP escalating to Fireworks and answer every remaining task
# with the local model only: a possibly-weaker local answer always beats a
# TIMEOUT, which zeros the whole submission. 510s leaves 90s of margin
# inside the 10-minute hard cap.
TIME_BUDGET_SECONDS = float(os.environ.get("TIME_BUDGET_SECONDS", "510"))


def _fallback_answer(prompt: str, router: Router) -> str:
    """Best-effort local answer used when a task raises unexpectedly."""
    try:
        return router._local_answer(prompt)
    except Exception:
        return "Unable to produce an answer for this task."


def main() -> int:
    start = time.time()

    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as fh:
            tasks = json.load(fh)
        assert isinstance(tasks, list)
    except Exception as exc:
        print(f"FATAL: cannot read tasks from {INPUT_PATH}: {exc}", file=sys.stderr)
        return 1

    local_model = get_local_model()
    print(f"local model backend: {local_model.backend}", flush=True)
    router = Router(local_model, FireworksClient())

    results = []
    budget_hit = False
    for task in tasks:
        task_id = task.get("task_id", "")
        prompt = task.get("prompt", "")
        try:
            if time.time() - start > TIME_BUDGET_SECONDS:
                if not budget_hit:
                    budget_hit = True
                    print(
                        f"TIME BUDGET {TIME_BUDGET_SECONDS:.0f}s exceeded — "
                        "answering all remaining tasks locally (no more API calls)",
                        flush=True,
                    )
                answer, meta = _fallback_answer(prompt, router), {"route": "deadline_local"}
            else:
                answer, meta = router.route(prompt)
        except Exception as exc:  # one bad task must never sink the run
            answer, meta = _fallback_answer(prompt, router), {"route": "error", "error": str(exc)}
        if not isinstance(answer, str) or not answer.strip():
            answer = "No answer available."
        results.append({"task_id": task_id, "answer": answer})
        print(f"[{task_id}] route={meta.get('route')} model={meta.get('model', '-')}", flush=True)

    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)

    elapsed = time.time() - start
    print(
        f"done: {len(results)} tasks in {elapsed:.1f}s | "
        f"fireworks calls={router.fireworks.calls} tokens={router.fireworks.total_tokens}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
