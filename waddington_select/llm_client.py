"""
llm_client.py — Provider-agnostic LLM client for the gene-selection arms.

The gene-selection code historically hard-coded an Anthropic client. This wraps that in a small
abstraction with three backends so the arm's own reasoning model can use any provider that
feynman/pi supports (Claude, OpenAI/codex, Gemini, …) without touching the arm logic.

Backends
--------
- "anthropic"  (default): direct Anthropic SDK — identical to the previous behaviour, so the
  paper benchmark stays reproducible on a fixed model.
- "pi": shell out to the feynman/pi CLI in one-shot mode (`-p`). This reuses feynman's provider
  routing and OAuth token handling (including openai-codex, whose auth is not a plain API key),
  giving true "any provider" without re-implementing each provider's auth in Python.
- "mock": deterministic echo for offline tests.

Selection & config (env, so no code change to switch providers)
---------------------------------------------------------------
- WADDINGTON_LLM_BACKEND   anthropic | pi | mock            (default: anthropic)
- WADDINGTON_PI_CMD        pi one-shot command              (default: "feynman --prompt")
                           if it contains "{prompt}" the prompt is substituted there; otherwise
                           the prompt is appended as the final argument.
- WADDINGTON_PI_MODEL      optional "provider/model" passed to pi via --model
- WADDINGTON_PI_STDIN      "1" to pass the prompt on stdin instead of as an argument

NOTE: the exact feynman one-shot invocation must be validated with a live token before the "pi"
backend is used for real runs; it is configurable above precisely so it can be tuned without
editing code.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

AUTH_JSON = Path.home() / ".feynman" / "agent" / "auth.json"

DEFAULT_BACKEND = os.environ.get("WADDINGTON_LLM_BACKEND", "anthropic")
DEFAULT_PI_CMD = os.environ.get("WADDINGTON_PI_CMD", "feynman --prompt")

# Retry schedule (mirrors the arm's previous backoff).
_MAX_ATTEMPTS = 8
_BASE_WAIT = 10
_MAX_WAIT = 120


class LLMError(Exception):
    """Raised when an LLM call fails after all retries, regardless of backend."""


def _load_anthropic_token() -> str:
    with open(AUTH_JSON) as f:
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
        anthropic backend; the pi backend delegates sampling to feynman's own settings).
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
            return self._complete_pi(prompt)
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

    def _complete_pi(self, prompt: str) -> str:
        cmd = shlex.split(DEFAULT_PI_CMD)
        pi_model = os.environ.get("WADDINGTON_PI_MODEL")
        if pi_model:
            cmd += ["--model", pi_model]
        use_stdin = os.environ.get("WADDINGTON_PI_STDIN") == "1"

        if "{prompt}" in DEFAULT_PI_CMD:
            cmd = [c.replace("{prompt}", prompt) for c in cmd]
            stdin_data = None
        elif use_stdin:
            stdin_data = prompt
        else:
            cmd = cmd + [prompt]
            stdin_data = None

        run_kwargs: dict = {"capture_output": True, "text": True, "timeout": 300}
        if stdin_data is not None:
            run_kwargs["input"] = stdin_data
        else:
            # Never inherit the parent's stdin — feynman would block waiting on the TTY.
            run_kwargs["stdin"] = subprocess.DEVNULL
        try:
            proc = subprocess.run(cmd, **run_kwargs)
        except (subprocess.TimeoutExpired, OSError) as e:
            raise _PiCallError(str(e)) from e
        if proc.returncode != 0:
            raise _PiCallError(f"exit {proc.returncode}: {proc.stderr.strip()[:200]}")
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
