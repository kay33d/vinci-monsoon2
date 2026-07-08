# Hybrid Router — AMD Hackathon ACT II, Track 1

Token-efficient general-purpose agent: a **two-stage local-first router** that
answers as much as possible with a bundled local GGUF model (0 scored tokens)
and escalates only hard tasks to Fireworks through the judging proxy.

## Architecture

```
/input/tasks.json
      │
      ▼
Stage 1 — classify LOCALLY (llama.cpp, GBNF grammar-constrained JSON)
      {"intent": <8 categories>, "difficulty": shallow|deep, "confidence": high|low}
      │
      ▼
Stage 2 — per-category escalation policy (config/routing_map.yaml)
      ├─ trusted-shallow  → answer with the LOCAL model        (0 tokens)
      └─ otherwise        → category → role → ALLOWED_MODELS   (Fireworks)
                              reasoning: math, logic
                              code:      debugging, generation
                              general:   factual, sentiment, summarize, NER
      │  (remote failure → local fallback: never an empty answer)
      ▼
/output/results.json  →  exit 0
```

Model IDs are **never hardcoded**: roles resolve at runtime by case-insensitive
substring match against the `ALLOWED_MODELS` env var, with graceful fallback to
its first entry. Edit the hints in [config/routing_map.yaml](config/routing_map.yaml)
on launch day.

## Repo layout

| Path | Purpose |
| --- | --- |
| `entrypoint.py` | Harness contract: read `/input/tasks.json`, write `/output/results.json` |
| `src/router/` | Stage-1 classifier + stage-2 dispatch/escalation |
| `src/local_models/loader.py` | CPU-only GGUF loader, grammar-constrained decode, heuristic dev fallback |
| `src/api_clients/fireworks.py` | Env-driven Fireworks client (retry, 25s timeout, token accounting) |
| `config/` | Routing map, escalation policy, prompt templates |
| `tests/` | Mock tasks, offline eval (Fireworks mocked), pytest schema tests |

## Environment variables (injected by the harness)

| Variable | Use |
| --- | --- |
| `FIREWORKS_API_KEY` | Auth for every Fireworks call |
| `FIREWORKS_BASE_URL` | ALL Fireworks traffic goes through this proxy |
| `ALLOWED_MODELS` | Comma-separated permitted model IDs, resolved at runtime |
| `LOCAL_MODEL_PATH` | GGUF path (default `/models/model.gguf`) |
| `INPUT_PATH` / `OUTPUT_PATH` | Local-dev overrides of the harness paths |

## Local development

```bash
pip install -r requirements-dev.txt
make test                       # offline eval + schema tests (no key, no model)
# or:
python tests/run_eval.py
python -m pytest tests/ -q
```

Without a GGUF present, the loader drops to a keyword-heuristic backend so the
pipeline stays runnable offline. Bundle/point to a real model via
`LOCAL_MODEL_PATH` to exercise true local inference.

## Build & submit

```bash
docker build --platform linux/amd64 -t <registry>/<user>/hybrid-router:latest .
docker push <registry>/<user>/hybrid-router:latest
```

Smoke test exactly like the harness:

```bash
docker run --rm \
  -v "$PWD/tests/mock_tasks.json:/input/tasks.json:ro" \
  -v "$PWD/out:/output" \
  -e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL -e ALLOWED_MODELS \
  <registry>/<user>/hybrid-router:latest
cat out/results.json
```

Default bundled model: **Qwen2.5-3B-Instruct Q4_K_M** (~2 GB), sized for the
4 GB RAM / 2 vCPU / no-GPU grading VM. Swap via
`--build-arg MODEL_URL=<gguf-url>` (e.g. a 1.5B for extra headroom).

## Compliance checklist (participant guide)

- [x] Reads `/input/tasks.json`, writes valid `/output/results.json`, exit 0
- [x] Env-driven key/base-url/models — nothing hardcoded, all calls via proxy
- [x] Per-request timeout 25s < 30s cap; soft 9-min deadline < 10-min cap
- [x] CPU-only linux/amd64 image, ~2.5 GB compressed ≪ 10 GB cap
- [x] No hardcoded/cached answers; per-task try/except, local fallback
- [x] English-only, concise outputs (per-category format instructions)
