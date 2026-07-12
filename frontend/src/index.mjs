// index.mjs — entry point. Auth → choose CLI or Web → tool-less conversation over the C-arm pipeline.

import { parseArgs } from "node:util";
import { stdin } from "node:process";

import { select, isCancel } from "@clack/prompts";

import { authorizedProviders, promptAuthorize, runSetup, pickDefaultChatModel } from "./setup.mjs";
import { runChat } from "./chat.mjs";
import { complete } from "./llm/complete.mjs";
import { launchWebServer } from "./web-server.mjs";
import { DATASETS } from "./brain.mjs";

function printHelp() {
  console.log(`Waddington — conversational gene selection (tool-less pi-ai shell over the C-arm pipeline).

Usage:
  waddington                 Interactive: authorize a provider, then pick Terminal (CLI) or Web UI
  waddington setup           Authorize model providers (Claude / Codex / Gemini)
  waddington --web [--port N]  Launch the Web UI directly (default port 3000)
  waddington complete        One-shot completion (reads prompt from stdin or --prompt); used by the
                             Python pipeline's provider-agnostic backend
  waddington --help

Options:
  --model provider/model     Conversation model (default: an authorized Claude Haiku)
  --port N                   Web UI port (default 3000)

Supported phenotypes: ${DATASETS.join(", ")}`);
}

async function readStdin() {
  let data = "";
  stdin.setEncoding("utf8");
  for await (const chunk of stdin) data += chunk;
  return data.trim();
}

/** One-shot tool-less completion — the hook the Python llm_client "pi" backend calls. */
async function oneShotComplete(values) {
  const prompt = values.prompt || (await readStdin());
  if (!prompt) throw new Error("complete: no prompt (pass --prompt or pipe on stdin)");
  const spec = values.model || pickDefaultChatModel();
  const out = await complete({
    spec,
    prompt,
    ...(values.system ? { systemPrompt: values.system } : {}),
    temperature: values.temperature != null ? Number(values.temperature) : 0,
    maxTokens: values["max-tokens"] != null ? Number(values["max-tokens"]) : 1200,
  });
  process.stdout.write(out + "\n");
}

export async function main() {
  const { values, positionals } = parseArgs({
    args: process.argv.slice(2),
    allowPositionals: true,
    options: {
      help: { type: "boolean" },
      web: { type: "boolean" },
      port: { type: "string" },
      model: { type: "string" },
      prompt: { type: "string" },
      system: { type: "string" },
      temperature: { type: "string" },
      "max-tokens": { type: "string" },
    },
  });

  const command = positionals[0];

  if (values.help || command === "help") return printHelp();
  if (command === "setup") return runSetup();
  if (command === "complete") return oneShotComplete(values);

  const port = values.port ? parseInt(values.port, 10) : 3000;

  if (values.web) {
    return launchWebServer({ port, modelSpec: values.model || pickDefaultChatModel() });
  }

  // Interactive: ensure at least one authorized provider, prompting to authorize if needed.
  let authorized = authorizedProviders();
  while (authorized.length === 0) {
    console.log("\nNo model provider is authorized yet — let's authorize one.\n");
    const ok = await promptAuthorize();
    if (!ok) {
      console.log("Authorization is required to continue.");
      return;
    }
    authorized = authorizedProviders();
  }
  console.log(`\nAuthorized providers: ${authorized.join(", ")}`);

  // Startup mode selector (interactive TTY only).
  if (stdin.isTTY) {
    const choice = await select({
      message: "How would you like to start?",
      initialValue: "cli",
      options: [
        { value: "cli", label: "Terminal  (CLI)", hint: "chat in this terminal" },
        { value: "web", label: "Web UI", hint: `browser at localhost:${port}` },
        { value: "auth", label: "Authorize another provider" },
      ],
    });
    if (isCancel(choice)) return;
    if (choice === "auth") {
      await promptAuthorize();
      return main();
    }
    if (choice === "web") {
      return launchWebServer({ port, modelSpec: values.model || pickDefaultChatModel() });
    }
  }

  return runChat({ modelSpec: values.model || pickDefaultChatModel() });
}
