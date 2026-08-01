// Tool-less pi-ai LLM bridge.
//
// This is the whole point of the "use pi, not feynman" decision: we get pi's unified,
// multi-provider LLM access (Claude / codex / gemini) WITH OAuth token auto-refresh, but WITHOUT
// the coding-agent's tools. It is a pure completion — `completeSimple(model, { messages })` with no
// `tools` field — so the model can only talk, never act. Gene selection is done by the deterministic
// Python C-arm pipeline (see ../brain.mjs), not by the LLM wielding tools.
//
// Auth goes through pi-coding-agent's AuthStorage, which refreshes expired OAuth tokens with locking
// — so the recurring "token expired" pain of the old raw-token path is gone.
//
// Store path resolution (portable, SELF-CONTAINED by default):
//   1. $WADDINGTON_AUTH_PATH if set                → explicit override
//   2. ~/.waddington/agent/auth.json               → Waddington's OWN store (default; made on first login)
//   • WADDINGTON_REUSE_FEYNMAN=1 opts into reusing ~/.feynman/agent/auth.json when feynman is installed
//     (developer convenience — authorize once for both). Not the default: Waddington authorizes its own
//     providers so it does not depend on feynman being present.

import { homedir } from "node:os";
import { join } from "node:path";
import { existsSync } from "node:fs";

import { AuthStorage, ModelRegistry } from "@earendil-works/pi-coding-agent";
import { completeSimple } from "@earendil-works/pi-ai/compat";

function resolveAuthPath() {
  if (process.env.WADDINGTON_AUTH_PATH) return process.env.WADDINGTON_AUTH_PATH;
  // Opt-in: reuse feynman's shared token store if it exists and the user asked for it.
  if (process.env.WADDINGTON_REUSE_FEYNMAN === "1") {
    const feynman = join(homedir(), ".feynman", "agent", "auth.json");
    if (existsSync(feynman)) return feynman;
  }
  // Self-contained default: Waddington's own OAuth store, created on first `waddington setup` / login.
  return join(homedir(), ".waddington", "agent", "auth.json");
}

export const DEFAULT_AUTH_PATH = resolveAuthPath();

let _registry = null;

/** Lazily build a ModelRegistry backed by Waddington's OAuth store (see resolveAuthPath). */
export function getRegistry(authPath = DEFAULT_AUTH_PATH) {
  if (!_registry) {
    const auth = AuthStorage.create(authPath);
    _registry = ModelRegistry.create(auth);
  }
  return _registry;
}

/** Models that currently have working auth configured (fast; does not refresh tokens). */
export function listAvailableModels(authPath = DEFAULT_AUTH_PATH) {
  return getRegistry(authPath)
    .getAvailable()
    .map((m) => ({ provider: m.provider, id: m.id, name: m.name }));
}

/** Extract plain text from a pi AssistantMessage. */
function textOf(message) {
  return (message.content || [])
    .filter((c) => c.type === "text")
    .map((c) => c.text)
    .join("")
    .trim();
}

/**
 * One tool-less completion.
 *
 * @param {object} opts
 * @param {string} [opts.provider]     provider id, e.g. "anthropic" | "openai-codex" | "google"
 * @param {string} [opts.modelId]      model id within that provider
 * @param {string} [opts.spec]         "provider/model" convenience; overrides provider+modelId
 * @param {string} opts.prompt         user prompt
 * @param {string} [opts.systemPrompt] optional system prompt
 * @param {number} [opts.temperature]
 * @param {number} [opts.maxTokens]
 * @param {string} [opts.authPath]
 * @returns {Promise<string>} the model's text response
 */
export async function complete(opts) {
  const {
    prompt,
    systemPrompt,
    temperature,
    maxTokens = 1200,
    authPath = DEFAULT_AUTH_PATH,
  } = opts;

  let { provider, modelId } = opts;
  if (opts.spec) {
    const slash = opts.spec.indexOf("/");
    if (slash === -1) throw new Error(`model spec must be "provider/model", got: ${opts.spec}`);
    provider = opts.spec.slice(0, slash);
    modelId = opts.spec.slice(slash + 1);
  }
  if (!provider || !modelId) throw new Error("complete() needs provider+modelId (or spec)");
  if (!prompt) throw new Error("complete() needs a prompt");

  const registry = getRegistry(authPath);
  const model = registry.find(provider, modelId);
  if (!model) {
    const avail = listAvailableModels(authPath)
      .map((m) => `${m.provider}/${m.id}`)
      .join(", ");
    throw new Error(`unknown model ${provider}/${modelId}. Available: ${avail || "(none authorized)"}`);
  }

  const resolved = await registry.getApiKeyAndHeaders(model);
  if (!resolved.ok) {
    throw new Error(`no working auth for ${provider}: ${JSON.stringify(resolved)}`);
  }

  const message = await completeSimple(
    model,
    {
      ...(systemPrompt ? { systemPrompt } : {}),
      messages: [{ role: "user", content: prompt }],
      // NO `tools` field → tool-less by construction.
    },
    {
      ...(resolved.apiKey ? { apiKey: resolved.apiKey } : {}),
      ...(resolved.headers ? { headers: resolved.headers } : {}),
      ...(temperature != null ? { temperature } : {}),
      ...(maxTokens != null ? { maxTokens } : {}),
    },
  );

  if (message.stopReason === "error" || message.errorMessage) {
    throw new Error(`${provider}/${modelId} failed: ${message.errorMessage || "provider error"}`);
  }
  const text = textOf(message);
  if (!text) throw new Error(`empty response from ${provider}/${modelId} (stopReason=${message.stopReason})`);
  return text;
}
