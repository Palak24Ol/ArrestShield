"""Minimal Groq chat transport, standard library only.

Shared by the honeypot and by declared training-data augmentation. This module is
never imported by detection code: no score, threshold, feature, or label in
ArrestShield is produced by an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    """Raised when an LLM call cannot be made or fails."""


@dataclass(frozen=True)
class GroqSettings:
    api_key_environment_variable: str = "GROQ_API_KEY"
    base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.8
    max_tokens: int = 220
    timeout_seconds: int = 30
    maximum_retries: int = 2

    @classmethod
    def from_config(cls, config: Mapping[str, Any], section: str = "llm") -> "GroqSettings":
        values = config.get(section) or {}
        defaults = cls()
        return cls(
            api_key_environment_variable=str(
                values.get("api_key_environment_variable", defaults.api_key_environment_variable)
            ),
            base_url=str(values.get("base_url", defaults.base_url)),
            model=str(values.get("model", defaults.model)),
            temperature=float(values.get("temperature", defaults.temperature)),
            max_tokens=int(values.get("max_tokens", defaults.max_tokens)),
            timeout_seconds=int(values.get("timeout_seconds", defaults.timeout_seconds)),
            maximum_retries=int(values.get("maximum_retries", defaults.maximum_retries)),
        )


def load_environment_file(path) -> None:
    """Load KEY=VALUE lines from a git-ignored .env without overriding real env vars."""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def groq_chat(
    messages: Sequence[Mapping[str, str]],
    settings: GroqSettings,
    response_format: Mapping[str, str] | None = None,
) -> str:
    """POST to Groq's chat completions endpoint and return the message content.

    The key is read from the environment at call time and is never stored on an
    object, logged, or written to disk.
    """
    api_key = os.environ.get(settings.api_key_environment_variable, "").strip()
    if not api_key:
        raise LLMError(
            f"{settings.api_key_environment_variable} is not set. "
            "Export it or place it in a git-ignored .env file."
        )
    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": list(messages),
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    if response_format:
        payload["response_format"] = dict(response_format)
    body = json.dumps(payload).encode("utf-8")

    last_error: Exception | None = None
    for attempt in range(settings.maximum_retries + 1):
        request = urllib.request.Request(
            settings.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # urllib's default agent is rejected by the edge in front of Groq
                # with a Cloudflare 1010, so identify the client explicitly.
                "User-Agent": "ArrestShield/1.0 (research; +https://github.com/local/arrestshield)",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            return str(parsed["choices"][0]["message"]["content"]).strip()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:400]
            last_error = LLMError(f"Groq HTTP {error.code}: {detail}")
            if error.code in (400, 401, 403, 404):
                raise last_error from error
            if error.code == 429 and attempt < settings.maximum_retries:
                time.sleep(5.0 * (attempt + 1))
                continue
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as error:
            last_error = LLMError(f"Groq request failed: {error}")
        if attempt < settings.maximum_retries:
            time.sleep(1.5 * (attempt + 1))
    raise last_error or LLMError("Groq request failed")
