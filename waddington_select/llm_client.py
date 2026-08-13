"""
llm_client.py — Provider-agnostic LLM client for the gene-selection arms.

The gene-selection code historically hard-coded an Anthropic client. This wraps that in a small
abstraction with three backends so the arm's own reasoning model can use any provider that
pi-ai supports (OpenAI/Codex, Claude, Gemini, …) without touching the arm logic.

Backends
--------
- "pi" (default): shell out to the frontend's tool-less pi-ai bridge (`node frontend/bin/waddington.js
  complete`). This reuses pi-ai's multi-provider routing + OAuth token handling *with auto-refresh*
  (incl. openai-codex, whose auth is not a plain API key), giving true "any provider" without
  re-implementing each provider's auth in Python. Unlike the old `feynman --prompt` path, the bridge
  is a PURE completion (no tools, no file writes), so it needs no cwd sandbox and cannot pollute the
  repo. The refresh also fixes the recurring "anthropic token expired" failures of the raw-token
  path above.
- "anthropic" (legacy reproduction only): direct Anthropic SDK — identical to the historical
  paper behaviour when explicitly selected by the frozen reproduction scripts.
- "mock": deterministic echo for offline tests.

Selection & config (env, so no code change to switch providers)
---------------------------------------------------------------
- WADDINGTON_LLM_BACKEND   pi | anthropic | mock            (default: pi)
- WADDINGTON_PI_CMD        completion command               (default: "node <repo>/frontend/bin/waddington.js complete")
                           if it contains "{prompt}" the prompt is substituted there; otherwise the
                           prompt is passed on stdin (WADDINGTON_PI_STDIN defaults to "1").
- WADDINGTON_PI_MODEL      "provider/model" passed to the bridge
                           (default: openai-codex/gpt-5.6-luna)
- WADDINGTON_PI_REASONING  Codex reasoning effort           (default: none)
- WADDINGTON_PI_STDIN      "1" (default) to pass the prompt on stdin; "0" to append it as an argument
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_BIN = REPO_ROOT / "frontend" / "bin" / "waddington.js"


def resolve_auth_json() -> Path:
    """The OAuth/token store path, self-contained by default (mirrors the frontend bridge).

    ``WADDINGTON_AUTH_PATH`` overrides; ``WADDINGTON_REUSE_FEYNMAN=1`` forces feynman's shared store;
    otherwise prefer Waddington's own ``~/.waddington`` store, falling back to ``~/.feynman`` for existing
    setups. Authorize a provider with ``node frontend/bin/waddington.js setup`` (OAuth into ~/.waddington).
    """
    override = os.environ.get("WADDINGTON_AUTH_PATH")
    if override:
        return Path(override)
    own = Path.home() / ".waddington" / "agent" / "auth.json"
    feynman = Path.home() / ".feynman" / "agent" / "auth.json"
    if os.environ.get("WADDINGTON_REUSE_FEYNMAN") == "1" and feynman.exists():
        return feynman
    if own.exists():
        return own
    if feynman.exists():
        return feynman
    return own

LEGACY_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_PI_MODEL = os.environ.get("WADDINGTON_PI_MODEL", "openai-codex/gpt-5.6-luna")
DEFAULT_PI_REASONING = os.environ.get("WADDINGTON_PI_REASONING", "none")
DEFAULT_BACKEND = os.environ.get("WADDINGTON_LLM_BACKEND", "pi")
# The "pi" backend now calls the frontend's tool-less pi-ai bridge (a pure completion), not the old
# `feynman --prompt` full agent. The bridge writes nothing, so no cwd sandbox is needed.
DEFAULT_PI_CMD = os.environ.get("WADDINGTON_PI_CMD", f"node {FRONTEND_BIN} complete")


def default_model_for_backend(backend: str | None = None) -> str:
    """Return the active model identifier without conflating GPT deployment and Claude history."""
    selected = backend or DEFAULT_BACKEND
    if selected == "anthropic":
        return os.environ.get("WADDINGTON_LLM_MODEL", LEGACY_ANTHROPIC_MODEL)
    return os.environ.get("WADDINGTON_LLM_MODEL", DEFAULT_PI_MODEL)

# Retry schedule (mirrors the arm's previous backoff).
_MAX_ATTEMPTS = 8
_BASE_WAIT = 10
_MAX_WAIT = 120


class LLMError(Exception):
    """Raised when an LLM call fails after all retries, regardless of backend."""


def _load_anthropic_token() -> str:
    with open(resolve_auth_json()) as f:
        return json.load(f)["anthropic"]["access"]


class LLMClient:
    """One LLM per (model, temperature); .complete(prompt) returns the response text."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        backend: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backend = backend or DEFAULT_BACKEND
        self._anthropic = None
        if self.backend == "anthropic":
            import anthropic

            self._anthropic = anthropic.Anthropic(auth_token=_load_anthropic_token())

    # -- public API --------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's text response, retrying transient failures with backoff.

        temperature/max_tokens override the per-client defaults for this call (honoured by the
        anthropic backend; the pi backend delegates sampling to the selected pi-ai provider/model).
        """
        wait = _BASE_WAIT
        last_err: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return self._complete_once(prompt, temperature, max_tokens)
            except _RETRYABLE as e:
                last_err = e
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                print(f"    [LLM/{self.backend}] {type(e).__name__} "
                      f"(attempt {attempt + 1}/{_MAX_ATTEMPTS}), waiting {wait}s...", flush=True)
                time.sleep(wait)
                wait = min(wait * 2, _MAX_WAIT)
        raise LLMError(f"{self.backend} backend failed after {_MAX_ATTEMPTS} attempts: {last_err}")

    # -- backends ----------------------------------------------------------------

    def _complete_once(self, prompt: str, temperature=None, max_tokens=None) -> str:
        temp = self.temperature if temperature is None else temperature
        mt = self.max_tokens if max_tokens is None else max_tokens
        if self.backend == "anthropic":
            return self._complete_anthropic(prompt, temp, mt)
        if self.backend == "pi":
            return self._complete_pi(prompt, temp, mt)
        if self.backend == "mock":
            return _MOCK_RESPONDER(prompt)
        raise LLMError(f"unknown backend '{self.backend}'")

    def _complete_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> str:
        resp = self._anthropic.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    def _complete_pi(self, prompt: str, temperature: float, max_tokens: int) -> str:
        cmd = shlex.split(DEFAULT_PI_CMD)
        # Honour an explicitly constructed client's model unless the deployment-level pi model
        # override is set.  Most arms use the shared default, but tests and downstream callers may
        # deliberately instantiate LLMClient with another provider/model spec.
        pi_model = os.environ.get("WADDINGTON_PI_MODEL", self.model or DEFAULT_PI_MODEL)
        cmd += [
            "--model", pi_model,
            "--temperature", str(temperature),
            "--max-tokens", str(max_tokens),
        ]
        reasoning = os.environ.get("WADDINGTON_PI_REASONING", DEFAULT_PI_REASONING)
        if reasoning:
            cmd += ["--reasoning", reasoning]
        # The bridge reads the prompt from stdin by default (safe for large prompts, no ARG_MAX limit).
        use_stdin = os.environ.get("WADDINGTON_PI_STDIN", "1") == "1"

        if "{prompt}" in DEFAULT_PI_CMD:
            cmd = [c.replace("{prompt}", prompt) for c in cmd]
            stdin_data = None
        elif use_stdin:
            stdin_data = prompt
        else:
            cmd = cmd + ["--prompt", prompt]
            stdin_data = None

        run_kwargs: dict = {"capture_output": True, "text": True, "timeout": 300}
        if stdin_data is not None:
            run_kwargs["input"] = stdin_data
        else:
            run_kwargs["stdin"] = subprocess.DEVNULL
        try:
            proc = subprocess.run(cmd, **run_kwargs)
        except (subprocess.TimeoutExpired, OSError) as e:
            raise _PiCallError(str(e)) from e
        if proc.returncode != 0:
            detail = proc.stderr.strip()[:500]
            # Configuration/auth/quota errors do not improve with an exponential retry loop.  Stop
            # immediately so a long experiment can be resumed after the operator fixes the cause.
            if any(marker in detail.lower() for marker in (
                "unsupported parameter", "unknown model", "no working auth", "not authorized",
                "insufficient_quota", "usage limit", "out of budget", "quota exceeded",
            )):
                raise LLMError(f"pi backend cannot continue: exit {proc.returncode}: {detail}")
            raise _PiCallError(f"exit {proc.returncode}: {detail}")
        out = proc.stdout.strip()
        if not out:
            raise _PiCallError("empty response from pi")
        return out


# ---------------------------------------------------------------------------
# Retryable-exception set (anthropic errors added lazily so import stays cheap)
# ---------------------------------------------------------------------------

class _PiCallError(Exception):
    """Transient pi subprocess failure — retried like a rate limit."""


def _retryable_exceptions() -> tuple[type, ...]:
    exc: list[type] = [_PiCallError]
    try:
        import anthropic

        exc += [
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.OverloadedError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
        ]
    except Exception:
        pass
    return tuple(exc)


_RETRYABLE = _retryable_exceptions()


# ---------------------------------------------------------------------------
# Mock backend (offline tests)
# ---------------------------------------------------------------------------

def _default_mock(prompt: str) -> str:
    """Deterministic stand-in: echoes a JSON gene list so parsers have something to chew on."""
    return '["TP53", "EGFR", "MYC", "KRAS"]'


_MOCK_RESPONDER = _default_mock


def set_mock_responder(fn) -> None:
    """Override the mock backend's responder in tests."""
    global _MOCK_RESPONDER
    _MOCK_RESPONDER = fn
