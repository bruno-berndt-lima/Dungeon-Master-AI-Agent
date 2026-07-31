"""The single provider boundary.

Every agent gets its client from `create_llm(agent_type)`. Change the model, the
host, or the provider here — never in an agent.

Models are resolved per agent role, because the roles have different needs: the
supervisor runs on every turn and only classifies, so latency dominates; the
narrative agents are read by a human, so quality shows. Both are overridable by
environment variable, so a different machine can pick different models without
touching this file:

    DND_MODEL_SUPERVISOR=qwen2.5:7b   # one role
    DND_MODEL_DEFAULT=llama3.2:3b     # every role that has no specific override
    OLLAMA_HOST=http://box.local:11434
"""

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from langchain_ollama import ChatOllama

# Ollama tags are lowercase and carry a size suffix. A bare "llama3.2" resolves
# to the latest tag; the capitalised name this module used to default to 404s.
DEFAULT_MODEL = "llama3.2:3b"

AGENT_MODELS = {
    "supervisor": "llama3.2:3b",      # every turn, pure classification
    "dice_roller": "llama3.2:3b",     # extraction into a fixed schema
    "researcher": "qwen2.5:7b",       # grounded answers over retrieved text
    "dungeon_master": "qwen2.5:7b",   # narrative coherence
}

DEFAULT_HOST = "http://localhost:11434"

ENV_MODEL_DEFAULT = "DND_MODEL_DEFAULT"
ENV_MODEL_PREFIX = "DND_MODEL_"


class OllamaUnavailableError(RuntimeError):
    """The daemon is unreachable, or it does not have the requested model."""


def resolve_host() -> str:
    """The Ollama base URL, honouring `OLLAMA_HOST`.

    Ollama's own CLI accepts a bare `host:port`, so normalise that to a URL.
    """
    host = os.environ.get("OLLAMA_HOST", "").strip() or DEFAULT_HOST
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def resolve_model(agent_type: Optional[str] = None) -> str:
    """Pick the model for an agent role.

    Precedence: `DND_MODEL_<AGENT_TYPE>` → `DND_MODEL_DEFAULT` → the per-agent
    map → `DEFAULT_MODEL`.
    """
    if agent_type:
        specific = os.environ.get(f"{ENV_MODEL_PREFIX}{agent_type.upper()}", "").strip()
        if specific:
            return specific

    fallback = os.environ.get(ENV_MODEL_DEFAULT, "").strip()
    if fallback:
        return fallback

    if agent_type and agent_type in AGENT_MODELS:
        return AGENT_MODELS[agent_type]

    return DEFAULT_MODEL


def list_installed_models(host: Optional[str] = None, timeout: float = 2.0) -> list[str]:
    """Tags installed on the daemon. Raises `OllamaUnavailableError` if it is down."""
    host = host or resolve_host()
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise OllamaUnavailableError(_host_message(host)) from exc
    return [model.get("name", "") for model in payload.get("models", [])]


def _host_message(host: str) -> str:
    return (
        f"Cannot reach the Ollama daemon at {host}. Start it with `ollama serve` "
        f"(or launch Ollama.app), or point OLLAMA_HOST at a running one."
    )


def _model_message(model: str, host: str) -> str:
    return (
        f"The Ollama daemon at {host} has no model tagged '{model}'. "
        f"Install it with `ollama pull {model}`, or override the model for this "
        f"agent with {ENV_MODEL_PREFIX}<AGENT_TYPE>."
    )


def _friendly_message(exc: BaseException, model: str, host: str) -> Optional[str]:
    """Translate an Ollama failure into something actionable, or None to re-raise.

    The client library raises `httpx.ConnectError` for a dead daemon and
    `ollama.ResponseError` with a 404 for a missing model. Neither is imported
    here — matching on type name and text keeps this independent of the client's
    exception hierarchy, which has moved before.
    """
    text = str(exc).lower()
    name = type(exc).__name__

    if isinstance(exc, (ConnectionError, TimeoutError)) or name in {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "RemoteProtocolError",
    }:
        return _host_message(host)

    if "connection" in text and ("refused" in text or "error" in text):
        return _host_message(host)

    if "not found" in text and (model.lower() in text or "model" in text):
        return _model_message(model, host)

    return None


class OllamaChat(ChatOllama):
    """`ChatOllama` that reports daemon and model problems in plain language.

    The two failures this project hits constantly — daemon not running, model
    not pulled — surface from the client as a bare `ConnectError` or a 404
    `ResponseError`. Both are actionable, and neither says so. Every entry point
    the agents use (direct `.invoke`, LCEL pipes, and the streaming path PR-06
    needs) is wrapped.
    """

    def _translate(self, exc: BaseException) -> BaseException:
        message = _friendly_message(exc, self.model, self.base_url or resolve_host())
        if message is None:
            return exc
        return OllamaUnavailableError(message)

    def invoke(self, *args, **kwargs):
        try:
            return super().invoke(*args, **kwargs)
        except Exception as exc:
            translated = self._translate(exc)
            if translated is exc:
                raise
            raise translated from exc

    async def ainvoke(self, *args, **kwargs):
        try:
            return await super().ainvoke(*args, **kwargs)
        except Exception as exc:
            translated = self._translate(exc)
            if translated is exc:
                raise
            raise translated from exc

    def stream(self, *args, **kwargs):
        try:
            yield from super().stream(*args, **kwargs)
        except Exception as exc:
            translated = self._translate(exc)
            if translated is exc:
                raise
            raise translated from exc

    async def astream(self, *args, **kwargs):
        try:
            async for chunk in super().astream(*args, **kwargs):
                yield chunk
        except Exception as exc:
            translated = self._translate(exc)
            if translated is exc:
                raise
            raise translated from exc


def create_llm(
    agent_type: Optional[str] = None,
    temperature: float = 0,
    model: Optional[str] = None,
    **kwargs,
) -> OllamaChat:
    """Build the chat client for an agent role.

    Construction does not touch the network — the graph, and the test suite,
    build every agent before any daemon is needed. Problems are reported on the
    first call instead, by `OllamaChat`.
    """
    return OllamaChat(
        model=model or resolve_model(agent_type),
        temperature=temperature,
        base_url=resolve_host(),
        **kwargs,
    )
