"""Any OpenAI-compatible chat endpoint (Gemini's free tier, Ollama, Groq, ...) using only the standard library.

The user supplies their own key with `python -m companion key set` (stored outside the repository, see
keystore.py) or with environment variables, which win:
  GEMINI_API_KEY (or COMPANION_LLM_API_KEY)   key; GEMINI_API_KEY selects Google's endpoint by default
  COMPANION_LLM_BASE_URL                      e.g. http://localhost:11434/v1 for Ollama
  COMPANION_LLM_MODEL                         model name (see `python -m companion models`)
"""

import json
import os
import urllib.error
import urllib.request
from types import SimpleNamespace as NS

from . import keystore

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-2.5-flash"
TIMEOUT = 90


def settings() -> NS | None:
    """Endpoint settings (environment first, then `companion key set`), or None when nothing is configured."""
    stored = keystore.load()
    stored = stored if stored.get("provider") in ("gemini", "custom") else {}
    key = (os.environ.get("COMPANION_LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
           or os.environ.get("GOOGLE_API_KEY") or stored.get("api_key"))
    url = os.environ.get("COMPANION_LLM_BASE_URL") or stored.get("base_url") or (GEMINI_URL if key else None)
    if not url:
        return None
    model = os.environ.get("COMPANION_LLM_MODEL") or stored.get("model") or DEFAULT_MODEL
    return NS(key=key or "", url=url.rstrip("/"), model=model)


def post(url: str, headers: dict, payload: dict | None) -> dict:
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", **headers},
                                     method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:400]
        hint = " (free-tier rate limit; wait a minute)" if error.code == 429 else ""
        raise RuntimeError(f"LLM request failed: HTTP {error.code}{hint} {detail}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"LLM request failed: {error.reason}") from None


def list_models(config: NS | None = None, http=None) -> list[str]:
    config = config or settings()
    if config is None:
        raise RuntimeError("No LLM configured: set GEMINI_API_KEY or COMPANION_LLM_BASE_URL")
    data = (http or post)(f"{config.url}/models", _headers(config), None)
    return sorted(str(m.get("id", "")).removeprefix("models/") for m in data.get("data", []))


def _headers(config: NS) -> dict:
    return {"Authorization": f"Bearer {config.key}"} if config.key else {}


def complete(system: str, user: str, config: NS | None = None, http=None, max_tokens: int = 800) -> str:
    """One plain chat turn (used for non-agent answers)."""
    config = config or settings()
    data = (http or post)(f"{config.url}/chat/completions", _headers(config), {
        "model": config.model, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    return data["choices"][0]["message"].get("content") or ""


class OpenAICompatClient:
    """Presents the `client.messages.create(...)` shape that run_agent already uses."""

    def __init__(self, config: NS | None = None, http=None):
        self.config = config or settings()
        if self.config is None:
            raise RuntimeError("No LLM configured: set ANTHROPIC_API_KEY, GEMINI_API_KEY or COMPANION_LLM_BASE_URL")
        self.http = http or post  # resolved at call time so tests can swap it
        self.messages = self

    def create(self, model=None, max_tokens=1000, system="", tools=None, messages=()):
        payload = {"model": self.config.model, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, *convert_messages(messages)]}
        if tools:
            payload["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in tools]
        data = self.http(f"{self.config.url}/chat/completions", _headers(self.config), payload)
        return NS(content=parse_message(data["choices"][0]["message"]))


def convert_messages(messages) -> list[dict]:
    """Anthropic-style messages (text, tool_use, tool_result blocks) to OpenAI chat messages."""
    out = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
        elif message["role"] == "assistant":
            text = "".join(b.text for b in content if b.type == "text")
            calls = [{"id": b.id, "type": "function",
                      "function": {"name": b.name, "arguments": json.dumps(b.input)}} for b in content if b.type == "tool_use"]
            entry = {"role": "assistant", "content": text or None}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
        else:
            out.extend({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]}
                       for b in content if b.get("type") == "tool_result")
    return out


def parse_message(message: dict) -> list:
    blocks = []
    if message.get("content"):
        blocks.append(NS(type="text", text=message["content"]))
    for call in message.get("tool_calls") or []:
        try:
            arguments = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        blocks.append(NS(type="tool_use", id=call["id"], name=call["function"]["name"],
                         input=arguments if isinstance(arguments, dict) else {}))
    return blocks
