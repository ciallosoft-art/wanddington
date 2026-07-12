// setup.mjs — provider authorization.
//
// Auth goes through pi-coding-agent's AuthStorage (OAuth login + auto-refresh). The store path is
// resolved in complete.mjs (DEFAULT_AUTH_PATH): $WADDINGTON_AUTH_PATH, else feynman's shared store
// if present (authorize once for both), else a standalone ~/.waddington store created on login.

import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";

import { AuthStorage } from "@earendil-works/pi-coding-agent";
import { select, isCancel } from "@clack/prompts";

import { DEFAULT_AUTH_PATH, listAvailableModels } from "./llm/complete.mjs";

// Provider ids we surface, with friendly labels.
const PROVIDERS = [
  { id: "anthropic", label: "Claude (Anthropic)" },
  { id: "openai-codex", label: "Codex (OpenAI / ChatGPT)" },
  { id: "google-gemini-cli", label: "Gemini (Google)" },
];

export function makeAuth(authPath = DEFAULT_AUTH_PATH) {
  return AuthStorage.create(authPath);
}

/** Providers that currently have credentials in the store. */
export function authorizedProviders(authPath = DEFAULT_AUTH_PATH) {
  return makeAuth(authPath).list();
}

/** Prefer an anthropic haiku model; else any authorized model that isn't a known-unsupported one. */
export function pickDefaultChatModel(authPath = DEFAULT_AUTH_PATH) {
  if (process.env.WADDINGTON_CHAT_MODEL) return process.env.WADDINGTON_CHAT_MODEL;
  const avail = listAvailableModels(authPath);
  const haiku = avail.find((m) => m.provider === "anthropic" && m.id === "claude-haiku-4-5");
  if (haiku) return "anthropic/claude-haiku-4-5";
  const usable = avail.find((m) => !m.id.includes("spark"));
  return usable ? `${usable.provider}/${usable.id}` : "anthropic/claude-haiku-4-5";
}

function loginCallbacks() {
  const rl = createInterface({ input: stdin, output: stdout });
  return {
    rl,
    callbacks: {
      onAuth: (info) => {
        console.log(`\nOpen this URL in your browser to authorize:\n  ${info.url}`);
        if (info.instructions) console.log(info.instructions);
        console.log();
      },
      onDeviceCode: (info) => {
        console.log(`\nOpen: ${info.verificationUri}\nEnter code: ${info.userCode}\n`);
      },
      onPrompt: async (p) => (await rl.question(`${p.message}${p.placeholder ? ` (${p.placeholder})` : ""}: `)).trim(),
      onSelect: async (p) => {
        console.log(`\n${p.message}`);
        p.options.forEach((o, i) => console.log(`  ${i + 1}. ${o.label}`));
        const c = await rl.question(`Enter number (1-${p.options.length}): `);
        return p.options[parseInt(c, 10) - 1]?.id;
      },
      onProgress: (msg) => console.log(msg),
    },
  };
}

/** Run the OAuth login flow for one provider; persists to the shared auth store. */
export async function loginProvider(providerId, authPath = DEFAULT_AUTH_PATH) {
  const auth = makeAuth(authPath);
  const { rl, callbacks } = loginCallbacks();
  try {
    await auth.login(providerId, callbacks);
    console.log(`\n✓ Authorized ${providerId}.\n`);
  } finally {
    rl.close();
  }
}

/** Interactive: pick a provider and authorize it. Returns true if something was authorized. */
export async function promptAuthorize(authPath = DEFAULT_AUTH_PATH) {
  const already = new Set(authorizedProviders(authPath));
  const choice = await select({
    message: "Authorize a model provider",
    options: PROVIDERS.map((p) => ({
      value: p.id,
      label: p.label + (already.has(p.id) ? "  (already authorized)" : ""),
    })),
  });
  if (isCancel(choice)) return false;
  await loginProvider(String(choice), authPath);
  return true;
}

/** `waddington setup` — show status, then let the user authorize providers. */
export async function runSetup(authPath = DEFAULT_AUTH_PATH) {
  const authorized = authorizedProviders(authPath);
  console.log(`Authorized providers: ${authorized.length ? authorized.join(", ") : "(none)"}`);
  await promptAuthorize(authPath);
}
