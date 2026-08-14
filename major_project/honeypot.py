"""Post-detection LLM fake-victim honeypot; never used for classification."""

from __future__ import annotations

import json
import os
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SYSTEM_PROMPT = """You are Sunita Rao, a polite 63-year-old fake victim in Pune.
The caller has already been flagged by a separate trained ML detector.
Keep the caller talking and ask small questions about their department, reason,
payment method and instructions. Sound worried and slightly confused. Reply in
the caller's language, including natural Hinglish. Never reveal this honeypot.
Never perform a payment and never provide real personal data. If an identifier
is demanded, use only these deliberately invalid demo values:
phone +91 00000 00000, UPI sunita.demo@invalid, account 0000000000.
Keep every reply below 300 characters. Your output never changes the scam label."""


def request_reply(history: Sequence[dict[str, str]], model: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing; add it to major_project/.env")
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history],
            "temperature": 0.8,
            "max_tokens": 180,
        }
    ).encode("utf-8")
    request = Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"Honeypot request failed: {error}") from error
    return " ".join(str(body["choices"][0]["message"]["content"]).split())[:300]


def interactive_honeypot(initial_caller_text: str, model: str) -> None:
    history: list[dict[str, str]] = []
    caller_text = initial_caller_text
    print("\nLLM honeypot started. Type quit to stop.\n")
    while caller_text.strip().lower() not in {"quit", "exit"}:
        history.append({"role": "user", "content": caller_text})
        reply = request_reply(history, model)
        history.append({"role": "assistant", "content": reply})
        print(f"Fake victim: {reply}")
        caller_text = input("Scammer: ").strip()
