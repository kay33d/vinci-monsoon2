"""Schema tests: results must be a valid array of non-empty {task_id, answer}."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.run_eval import run_offline_eval  # noqa: E402


def test_results_schema(tmp_path):
    out_file = tmp_path / "results.json"
    run_offline_eval(output_path=out_file)

    with open(out_file, "r", encoding="utf-8") as fh:
        results = json.load(fh)

    assert isinstance(results, list) and results, "results must be a non-empty JSON array"
    for entry in results:
        assert isinstance(entry, dict)
        assert set(entry.keys()) == {"task_id", "answer"}
        assert isinstance(entry["task_id"], str) and entry["task_id"]
        assert isinstance(entry["answer"], str) and entry["answer"].strip(), (
            f"empty answer for {entry.get('task_id')}"
        )


def test_all_mock_tasks_answered(tmp_path):
    results = run_offline_eval(output_path=tmp_path / "results.json")
    with open(ROOT / "tests" / "mock_tasks.json", "r", encoding="utf-8") as fh:
        tasks = json.load(fh)
    assert {r["task_id"] for r in results} == {t["task_id"] for t in tasks}
