// index.mjs — entry point. Auth → choose CLI or Web → tool-less conversation over the C-arm pipeline.

import { parseArgs } from "node:util";
import { stdin } from "node:process";

import { select, isCancel } from "@clack/prompts";

import { authorizedProviders, promptAuthorize, runSetup, pickDefaultChatModel } from "./setup.mjs";
import { runChat } from "./chat.mjs";
import { complete } from "./llm/complete.mjs";
import { launchWebServer } from "./web-server.mjs";
import { getDatasets } from "./brain.mjs";
import { autoRun, autoRunUpload } from "./autorun.mjs";

async function printHelp() {
  const DATASETS = await getDatasets();
  console.log(`Waddington — conversational gene selection (tool-less pi-ai shell over the C-arm pipeline).

Usage:
  waddington                 Interactive: authorize a provider, then pick Terminal (CLI) or Web UI
  waddington setup           Authorize model providers (Claude / Codex / Gemini)
  waddington --web [--port N]  Launch the Web UI directly (default port 3000)
  waddington run [DATASET]   Retrospective one-click run: treat a completed screen's readout as ground
                             truth, auto-run the sequential campaign, print a live step-by-step timeline
                             and the final hit@R5 (e.g. "waddington run IFNG --rounds 5")
  waddington run --upload FILE --name X [--task "…"] [--modality KO|CRISPRi|CRISPRa] [--cell-line "…"]
                 [--curated-essential] [--tier non_essential|full_pool]
                             Same, for a BRAND-NEW screen: register it from the uploaded readout (builds
                             its features — no code change), then run. The design sets the DepMap features:
                             none for CRISPRa or a curated essential panel, pan-cancer only in K562
                             (modality omitted = unknown, treated as loss-of-function).
  waddington complete        One-shot completion (reads prompt from stdin or --prompt); used by the
                             Python pipeline's provider-agnostic backend
  waddington --help

Options:
  --model provider/model     Conversation model (default: openai-codex/gpt-5.6-luna)
  --reasoning LEVEL          Codex effort (default: none; options: none/minimal/low/medium/high/xhigh/max)
  --port N                   Web UI port (default 3000)
  --reason-from-structure    Hide gene names from the model; pick from structural features only
                             (network / pathway / essentiality). Removes the memorisation
                             dependence; best for loss-of-function screens and novel genes.

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
    ...(values.temperature != null ? { temperature: Number(values.temperature) } : {}),
    maxTokens: values["max-tokens"] != null ? Number(values["max-tokens"]) : 1200,
    ...(values.reasoning ? { reasoning: values.reasoning } : {}),
  });
  process.stdout.write(out + "\n");
}

/**
 * `waddington run [DATASET] [--rounds N]` — retrospective one-click evaluation with a live terminal
 * timeline. Uses a benchmark screen's own readout as a stand-in for a scientist's completed screen,
 * runs the sequential C-arm campaign against it, and prints each step as it happens.
 */
const DEPMAP_TEXT = { none: "off", pan: "pan-cancer only", pan_k562: "pan-cancer + K562" };

async function runAutoDemo(values, positionals) {
  const rounds = values.rounds ? parseInt(values.rounds, 10) : 5;
  const spec = values.model || pickDefaultChatModel();
  const bar = (r) => "█".repeat(Math.round(r * 30)).padEnd(30, "·");
  const onStep = (e) => {
    if (e.type === "upload") console.log(`📤 Uploaded readout for a NEW screen "${e.name}" (${e.bytes} bytes).`);
    else if (e.type === "anchors") console.log(`   anchors: ${e.anchors.join(", ")}`);
    else if (e.type === "registering") console.log(`   building ML features (registering "${e.name}")…`);
    else if (e.type === "registered") {
      console.log(`✅ Registered "${e.name}" — ${e.nGenes} genes (${e.nWithFeatures} with known features). Now rankable.`);
      if (e.design) console.log(`   design: ${e.design.modality}${e.design.cell_line ? ` in ${e.design.cell_line}` : ""} → DepMap ${DEPMAP_TEXT[e.depmap] || e.depmap}\n`);
    }
    else if (e.type === "already-registered") console.log(`   "${e.name}" is already registered.\n`);
    else if (e.type === "intent") console.log(`🧠 ${e.text}\n   readout: ${e.readout}\n`);
    else if (e.type === "round-open") console.log(`── Round ${e.round}/${e.rounds} ─────────────────────────────`);
    else if (e.type === "proposed") {
      const c = e.counts || {}, tf = e.transfer || {};
      const top = Object.entries(e.shap || {}).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k]) => k).join(", ");
      const prior = tf.zeroShot
        ? `prior: trained on ${tf.refScreens} reference screens, 0 in-experiment labels yet (zero-shot transfer)`
        : `prior: ${tf.refScreens} reference screens + ${tf.inExpLabels} in-experiment labels`;
      console.log(`  ⚙ C-arm proposed ${e.genes.length} genes  (ML ${c.ml_only ?? "?"} · LLM ${c.llm_only ?? "?"} · both ${c.both ?? "?"};  w_ml ${e.w_ml}/w_llm ${e.w_llm})`);
      console.log(`     ${prior}`);
      if (top) console.log(`     top ML features: ${top}`);
    } else if (e.type === "revealed") {
      console.log(`  ✓ revealed from readout: +${e.nHits} hits  →  ${bar(e.ratio)}  ${e.cumulative}/${e.total}  hit@R${e.round}=${e.ratio.toFixed(4)}\n`);
    } else if (e.type === "summary") {
      console.log(`✅ Done.  hit@R${e.rounds} = ${e.hit_at_r5.toFixed(4)}  (${e.cumulative}/${e.total} hits found)\n`);
    }
  };

  if (values.upload) {
    // Real production path: register a brand-new screen from an uploaded readout, then run it.
    const name = values.name || positionals[1];
    if (!name) throw new Error("run --upload needs --name <phenotype id>");
    console.log(`\n▶ Uploaded-screen run — registering "${name}" from ${values.upload}, then ranking it.\n`);
    await autoRunUpload({
      name, readoutFile: values.upload, rounds, modelSpec: spec,
      task: values.task, measurement: values.measurement,
      modality: values.modality, cellLine: values["cell-line"], curatedEss: Boolean(values["curated-essential"]),
      tier: values.tier === "full_pool" ? "full_pool" : "non_essential",
      anchors: values.anchors ? values.anchors.split(/[,\s]+/).filter(Boolean) : undefined,
      onStep,
    });
    return;
  }

  const dataset = positionals[1] || values.dataset || "IFNG";
  console.log(`\n▶ Retrospective run — treating ${dataset}'s readout as an uploaded completed screen.\n`);
  await autoRun({ dataset, rounds, modelSpec: spec, readoutFile: values.readout || undefined, onStep });
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
      reasoning: { type: "string" },
      "reason-from-structure": { type: "boolean" },
      dataset: { type: "string" },
      rounds: { type: "string" },
      readout: { type: "string" },
      upload: { type: "string" },
      name: { type: "string" },
      task: { type: "string" },
      measurement: { type: "string" },
      anchors: { type: "string" },
      modality: { type: "string" },
      "cell-line": { type: "string" },
      "curated-essential": { type: "boolean" },
      tier: { type: "string" },
    },
  });

  const command = positionals[0];

  // Keep interactive narration and every child Python C-arm on the same requested effort. Previously
  // this flag affected only the one-shot `complete` subcommand and was silently ignored by CLI/Web.
  if (values.reasoning) process.env.WADDINGTON_PI_REASONING = values.reasoning;

  if (values.help || command === "help") return printHelp();
  if (command === "setup") return runSetup();
  if (command === "complete") return oneShotComplete(values);
  if (command === "run") return runAutoDemo(values, positionals);

  // Opt-in: make the C-arm reason over anonymized structural features instead of gene names. Set here
  // so it flows to every `waddington_select.suggest` subprocess (run() spreads process.env). Off by
  // default; a deliberate robustness trade — see the reason-vs-recall ablation.
  if (values["reason-from-structure"]) {
    process.env.WADDINGTON_REASON_FROM_STRUCTURE = "1";
    console.log("• Reasoning from structure: gene names are hidden from the model; picks are driven by "
      + "network / pathway / essentiality features only.\n");
  }

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
