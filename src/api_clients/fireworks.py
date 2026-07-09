"""Fireworks client — OpenAI-compatible chat completions over HTTP.

Hard rules honoured here:
  * FIREWORKS_API_KEY / FIREWORKS_BASE_URL are read from the ENVIRONMENT only.
  * Every call goes through FIREWORKS_BASE_URL (the judging proxy).
  * One retry with backoff, 25s default timeout — under the 30s per-request cap.
"""

from __future__ import annotations

import os
import time

import requests


class FireworksError(RuntimeError):
    """Raised when a Fireworks call fails after its retry."""


class FireworksClient:
    RETRIES = 1          # one retry with backoff
    BACKOFF_SECONDS = 2.0

    def __init__(self):
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url = os.environ.get("FIREWORKS_BASE_URL", "").rstrip("/")
        self.total_tokens = 0   # prompt + completion tokens as reported by API
        self.calls = 0

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    def chat(self, model: str, system: str, user: str,
             max_tokens: int = 512, timeout: float = 25.0) -> str:
        if not self.configured:
            raise FireworksError("FIREWORKS_API_KEY / FIREWORKS_BASE_URL not set")

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Exception | str | None = None
        for attempt in range(self.RETRIES + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            except requests.RequestException as exc:
                # Network error / timeout: genuinely transient — retry once.
                last_err = exc
                if attempt < self.RETRIES:
                    time.sleep(self.BACKOFF_SECONDS)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                # Rate limit / server error: transient — retry once.
                last_err = f"HTTP {resp.status_code}"
                if attempt < self.RETRIES:
                    time.sleep(self.BACKOFF_SECONDS)
                continue
            if resp.status_code >= 400:
                # Auth / unknown model / bad request: permanent. Retrying
                # would waste another full timeout — fail fast, caller
                # falls back to the local model.
                raise FireworksError(f"HTTP {resp.status_code} (permanent, not retried)")

            try:
                data = resp.json()
                choice = data["choices"][0]
                text = (choice["message"].get("content") or "").strip()
                finish = choice.get("finish_reason")
            except Exception as exc:
                raise FireworksError(f"malformed response (not retried): {exc}")

            # Tokens were consumed even if the answer is unusable — count them.
            self.total_tokens += data.get("usage", {}).get("total_tokens", 0)

            if not text or finish == "length":
                # Hidden-reasoning models can burn the whole max_tokens budget
                # and return empty/truncated content with finish_reason=length.
                # A retry would just repeat it and waste another timeout —
                # fail IMMEDIATELY so the caller falls back locally.
                raise FireworksError(
                    f"unusable completion (finish_reason={finish}, "
                    f"empty={not text}) — not retried"
                )

            self.calls += 1
            return text
        raise FireworksError(f"Fireworks call failed after retry: {last_err}")
