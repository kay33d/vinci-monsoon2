# Offline eval + schema tests (no API key, no network, no GGUF required)
test:
	python tests/run_eval.py
	python -m pytest tests/ -q

docker-build:
	docker build --platform linux/amd64 -t hybrid-router:local .

# Run the container against the mock tasks exactly like the harness would
docker-run:
	docker run --rm \
		-v "$(CURDIR)/tests/mock_tasks.json:/input/tasks.json:ro" \
		-v "$(CURDIR)/out:/output" \
		-e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL -e ALLOWED_MODELS \
		hybrid-router:local
