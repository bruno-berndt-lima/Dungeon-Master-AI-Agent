"""Unit tests for the provider boundary in `src/models/llm.py`.

Nothing here touches a daemon: model and host resolution are pure functions of
the environment, and the error translation is tested by feeding it exceptions
directly. `create_llm` itself is offline by design — construction must not fail
on a machine with no Ollama running, or the graph could not be built in CI.
"""

import pytest

from src.models.llm import (
    AGENT_MODELS,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    OllamaChat,
    OllamaUnavailableError,
    _friendly_message,
    create_llm,
    resolve_host,
    resolve_model,
)

AGENT_TYPES = ["supervisor", "researcher", "dice_roller", "dungeon_master"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from an unconfigured environment."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("DND_MODEL_DEFAULT", raising=False)
    for agent in AGENT_TYPES:
        monkeypatch.delenv(f"DND_MODEL_{agent.upper()}", raising=False)


# --- model resolution -------------------------------------------------------

@pytest.mark.parametrize("agent_type", AGENT_TYPES)
def test_every_agent_resolves_to_its_mapped_model(agent_type):
    assert resolve_model(agent_type) == AGENT_MODELS[agent_type]


def test_unknown_agent_falls_back_to_the_default_model():
    assert resolve_model("cartographer") == DEFAULT_MODEL
    assert resolve_model(None) == DEFAULT_MODEL


@pytest.mark.parametrize("model", list(AGENT_MODELS.values()) + [DEFAULT_MODEL])
def test_every_configured_tag_is_lowercase_and_sized(model):
    """KNOWN_ISSUES #22: 'Llama3.2' 404s. Tags are lowercase and carry a size."""
    assert model == model.lower()
    assert ":" in model


def test_agent_specific_env_var_wins(monkeypatch):
    monkeypatch.setenv("DND_MODEL_SUPERVISOR", "phi4:latest")
    assert resolve_model("supervisor") == "phi4:latest"
    assert resolve_model("researcher") == AGENT_MODELS["researcher"]


def test_default_env_var_covers_every_unoverridden_role(monkeypatch):
    monkeypatch.setenv("DND_MODEL_DEFAULT", "phi4:latest")
    monkeypatch.setenv("DND_MODEL_RESEARCHER", "qwen2.5:14b")
    assert resolve_model("supervisor") == "phi4:latest"
    assert resolve_model("researcher") == "qwen2.5:14b"


def test_blank_env_var_is_ignored(monkeypatch):
    monkeypatch.setenv("DND_MODEL_SUPERVISOR", "   ")
    assert resolve_model("supervisor") == AGENT_MODELS["supervisor"]


# --- host resolution --------------------------------------------------------

def test_host_defaults_to_localhost():
    assert resolve_host() == DEFAULT_HOST


@pytest.mark.parametrize(
    "value,expected",
    [
        ("http://box.local:11434", "http://box.local:11434"),
        ("box.local:11434", "http://box.local:11434"),      # ollama's bare form
        ("http://box.local:11434/", "http://box.local:11434"),
        ("https://box.local", "https://box.local"),
    ],
)
def test_host_is_normalised_to_a_url(monkeypatch, value, expected):
    monkeypatch.setenv("OLLAMA_HOST", value)
    assert resolve_host() == expected


# --- construction is offline ------------------------------------------------

def test_create_llm_makes_no_network_call(monkeypatch):
    """The graph builds all four agents before any daemon is required."""
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")  # nothing listens here
    llm = create_llm("supervisor")
    assert isinstance(llm, OllamaChat)
    assert llm.model == AGENT_MODELS["supervisor"]
    assert llm.base_url == "http://127.0.0.1:1"


def test_explicit_model_overrides_the_map():
    assert create_llm("supervisor", model="phi4:latest").model == "phi4:latest"


def test_temperature_is_still_supported():
    assert create_llm("dungeon_master", temperature=0.8).temperature == 0.8


# --- error translation ------------------------------------------------------

def test_dead_daemon_names_the_host():
    message = _friendly_message(ConnectionError("[Errno 61] Connection refused"),
                                "llama3.2:3b", "http://box.local:11434")
    assert message is not None
    assert "http://box.local:11434" in message
    assert "ollama serve" in message


def test_missing_model_names_the_pull_command():
    exc = Exception("model 'llama3.2:3b' not found (status code: 404)")
    message = _friendly_message(exc, "llama3.2:3b", DEFAULT_HOST)
    assert message is not None
    assert "ollama pull llama3.2:3b" in message


def test_unrelated_errors_are_left_alone():
    """Anything we can't explain must propagate untouched, not be reworded."""
    assert _friendly_message(ValueError("bad prompt template"),
                             "llama3.2:3b", DEFAULT_HOST) is None


def test_invoke_translates_a_connection_failure(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    llm = create_llm("supervisor")

    def boom(*args, **kwargs):
        raise ConnectionError("[Errno 61] Connection refused")

    monkeypatch.setattr(type(llm).__mro__[1], "invoke", boom, raising=True)

    with pytest.raises(OllamaUnavailableError) as caught:
        llm.invoke("hello")
    assert "http://127.0.0.1:1" in str(caught.value)


def test_invoke_passes_other_errors_through(monkeypatch):
    llm = create_llm("supervisor")

    def boom(*args, **kwargs):
        raise ValueError("bad prompt template")

    monkeypatch.setattr(type(llm).__mro__[1], "invoke", boom, raising=True)

    with pytest.raises(ValueError, match="bad prompt template"):
        llm.invoke("hello")
