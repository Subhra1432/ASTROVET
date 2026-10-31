"""Optional Groq-generated plain-language explanations of fixed ASTROVET results."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
REQUEST_TIMEOUT_SECONDS = 20

EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": ["summary", "evidence", "limitations", "next_step"],
    "additionalProperties": False,
}


def generate_groq_explanation(
    analysis_summary: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Ask Groq to explain a small, identifier-free summary without changing vetting."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None, "Groq explanation is off. Set GROQ_API_KEY locally to enable it."

    model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
    request_body = {
        "model": model,
        "temperature": 0.1,
        "max_completion_tokens": 450,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Explain only the supplied ASTROVET analysis summary in plain language. "
                    "Do not change or reinterpret its score, calibration state, or review decision. "
                    "Never say that a planet is confirmed or rejected. Do not invent missing measurements, "
                    "causes, probabilities, or evidence. State when evidence is limited or unavailable. "
                    "Explain that the output is research triage and needs human review. Keep each field concise."
                ),
            },
            {"role": "user", "content": json.dumps(analysis_summary, separators=(",", ":"))},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "astrovet_plain_language_explanation",
                "strict": True,
                "schema": EXPLANATION_SCHEMA,
            },
        },
    }
    request = Request(
        GROQ_CHAT_COMPLETIONS_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 401:
            return None, "Groq rejected the API key. Check or replace GROQ_API_KEY."
        if exc.code == 429:
            return None, "Groq is temporarily rate-limited. ASTROVET's result is unchanged."
        return None, f"Groq could not provide an explanation (HTTP {exc.code}). ASTROVET's result is unchanged."
    except (TimeoutError, URLError, OSError):
        return None, "Groq could not be reached. ASTROVET's result is unchanged."
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, "Groq returned an unreadable response. ASTROVET's result is unchanged."

    try:
        content = response_data["choices"][0]["message"]["content"]
        explanation = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None, "Groq returned an incomplete explanation. ASTROVET's result is unchanged."

    required_text_fields = ("summary", "limitations", "next_step")
    if not all(isinstance(explanation.get(field), str) for field in required_text_fields):
        return None, "Groq returned an invalid explanation. ASTROVET's result is unchanged."
    evidence = explanation.get("evidence")
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        return None, "Groq returned an invalid explanation. ASTROVET's result is unchanged."

    cleaned = {
        "summary": explanation["summary"][:900],
        "evidence": [item[:300] for item in evidence[:5]],
        "limitations": explanation["limitations"][:600],
        "next_step": explanation["next_step"][:400],
    }
    return cleaned, "AI-generated explanation. ASTROVET's original score and review decision are unchanged."
