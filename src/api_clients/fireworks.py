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

        last_err: Exception | None = None
        for attempt in range(self.RETRIES + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                self.total_tokens += usage.get("total_tokens", 0)
                self.calls += 1
                if not text:
                    raise ValueError("empty completion")
                return text
            except Exception as exc:  # network, HTTP, schema — retry once
                last_err = exc
                if attempt < self.RETRIES:
                    time.sleep(self.BACKOFF_SECONDS * (attempt + 1))
        raise FireworksError(f"Fireworks call failed after retry: {last_err}")
