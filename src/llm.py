"""
Single Ollama client for the whole pipeline.

Every module that embeds text or talks to Llama 3.1 goes through here, so
model names and endpoints live in exactly one place.
"""

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OLLAMA = os.environ["OLLAMA_URL"]
EMBED_MODEL = os.environ["EMBED_MODEL"]
LLM_MODEL = os.environ["LLM_MODEL"]

TIMEOUT = 120   # generation on CPU is slow; embeddings return in well under this

if "//localhost" in OLLAMA:
    # Ollama binds IPv4 only. On Windows "localhost" resolves to ::1 first, so
    # every call pays ~2s waiting for that to fail before retrying IPv4 —
    # measured 2.10s vs 0.06s per request. One question makes ~9 calls, so
    # this is ~18s of pure waiting. Loud, because it looks like a slow model.
    print(f"WARNING: OLLAMA_URL={OLLAMA} — use 127.0.0.1 instead of localhost; "
          "see .env.example", file=sys.stderr)

# One pooled session: keep-alive removes a TCP handshake per request, and
# there are a lot of requests per question.
_session = requests.Session()


def embed(text: str) -> list[float]:
    r = _session.post(f"{OLLAMA}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text},
                      timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["embedding"]


def chat(messages: list[dict], *, temperature: float = 0.2,
         json_mode: bool = False) -> str:
    """Chat completion. `json_mode` constrains Ollama to emit valid JSON."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},   # factual QA: low creativity
    }
    if json_mode:
        payload["format"] = "json"

    r = _session.post(f"{OLLAMA}/api/chat", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"]


def chat_json(messages: list[dict], *, temperature: float = 0.0) -> dict | None:
    """Chat in JSON mode, parsed. Returns None if the model emits junk —
    callers are expected to have a non-LLM fallback rather than crash."""
    try:
        return json.loads(chat(messages, temperature=temperature, json_mode=True))
    except (json.JSONDecodeError, requests.RequestException, KeyError):
        return None
