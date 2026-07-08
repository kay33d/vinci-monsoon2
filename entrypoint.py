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

# Leave a safety margin inside the 10-minute wall clock: once we cross this
# many seconds we stop escalating and answer the remainder locally/fast.
SOFT_DEADLINE_SECONDS = 540


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
    for task in tasks:
        task_id = task.get("task_id", "")
        prompt = task.get("prompt", "")
        try:
            if time.time() - start > SOFT_DEADLINE_SECONDS:
                # Out of budget: no more remote calls, just answer locally.
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
