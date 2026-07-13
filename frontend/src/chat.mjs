// chat.mjs — the tool-less terminal conversation (thin renderer over converse.respond).

import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";

import { DEFAULT_CHAT_MODEL } from "./intent.mjs";
import { respond, getDatasets } from "./converse.mjs";

const banner = (datasets) => `Waddington — conversational gene selection (tool-less, C-arm pipeline).
Tell me a phenotype and I'll recommend genes to perturb; report results and I'll adapt.
Supported phenotypes: ${datasets.join(", ")}.
Type "exit" to quit.\n`;

function render(result) {
  if (result.kind === "chat") {
    console.log(`\n${result.text}\n`);
    return;
  }
  if (result.kind === "onboard") {
    if (result.stage === "anchors") {
      console.log(`\n[Onboarding · ${result.name}] Proposed anchor genes (${result.proposed.length}):`);
      console.log("  " + result.proposed.join(", "));
      console.log(`\n${result.text}\n`);
      return;
    }
    if (result.stage === "done") {
      console.log(`\n[Phenotype registered · ${result.name}]`);
      console.log(`  gene pool: ${result.n_genes} (${result.n_genes_with_known_features} with known features)`);
      console.log(`  anchors:   ${result.anchors.join(", ")}`);
      console.log(`\nIt's now rankable. Say e.g. "run a real experiment on ${result.name} with my own results".\n`);
      return;
    }
    console.log(`\n${result.text}\n`);
    return;
  }
  if (result.kind === "campaign") {
    if (result.stage === "await") { console.log(`\n${result.text}\n`); return; }
    if (result.stage === "awaiting_results") {
      if (result.text) { console.log(`\n${result.text}\n`); return; }
      console.log(`\n[Round ${result.round} committed · ${result.genes.length} genes tested]`);
      console.log("→ paste the path to your screen results file (MAGeCK gene_summary or Gene,Score CSV), or `stop`.\n");
      return;
    }
    if (result.stage === "proposed") {
      console.log(`\n[Experiment · ${result.dataset}] Round ${result.round}/${result.maxRounds} — proposed batch (${result.genes.length} genes):`);
      console.log("  " + result.genes.join(", "));
      console.log("\n→ type `commit` to test this batch (reveal the phenotype), or `stop` to end.\n");
      return;
    }
    if (result.stage === "revealed") {
      console.log(`\n[Round ${result.round} results] ${result.roundHits.length} hits: ${result.roundHits.join(", ") || "(none)"}`);
      console.log(`  cumulative ${result.cumulative}/${result.total} hits (${(result.ratio * 100).toFixed(1)}%)`);
      console.log(`\n[Round ${result.nextRound}] proposed batch (${result.next.length} genes):`);
      console.log("  " + result.next.join(", "));
      console.log("\n→ `commit` to continue, or `stop` to end.\n");
      return;
    }
    if (result.stage === "done") {
      console.log(`\n[Experiment complete · ${result.dataset}]`);
      if (result.roundHits?.length) console.log(`  final-round hits: ${result.roundHits.join(", ")}`);
      console.log(`  cumulative ${result.cumulative}/${result.total} hits (${(result.ratio * 100).toFixed(1)}%) over ${result.history.length} rounds`);
      for (const h of result.history) {
        console.log(`    round ${h.round}: +${h.hits.length} → ${h.cumulative} (${(h.ratio * 100).toFixed(1)}%)`);
      }
      if (result.tracePath) console.log(`  trace saved: ${result.tracePath}`);
      console.log();
      return;
    }
  }
  if (result.kind === "simulate") {
    console.log(`\nDemo campaign on "${result.dataset}":\n`);
    console.log(result.text);
    return;
  }
  // suggest
  const fb = result.feedback;
  console.log(
    `\nRecommended next batch for "${result.dataset}" ` +
      `(round ${result.info?.round ?? 1}, ${result.genes.length} genes, route=${result.info?.route}` +
      (fb.hits + fb.misses ? `, feedback ${fb.hits} hits/${fb.misses} non-hits` : ", cold start") +
      "):",
  );
  if (result.info?.unknown_genes?.length) {
    console.log(`  [note] ignored (not in pool): ${result.info.unknown_genes.join(", ")}`);
  }
  console.log("  " + result.genes.join(", "));
  if (result.narration) console.log(`\n${result.narration}`);
  console.log(`\n(Report which of these were hits and I'll adapt the next round.)\n`);
}

export async function runChat({ modelSpec = DEFAULT_CHAT_MODEL } = {}) {
  console.log(banner(await getDatasets()));
  const state = { dataset: null, hits: [], misses: [] };
  const rl = createInterface({ input: stdin, output: stdout });
  try {
    while (true) {
      let line;
      try {
        line = (await rl.question("you › ")).trim();
      } catch {
        break; // readline closed (EOF / Ctrl-D / piped input exhausted)
      }
      if (!line) continue;
      if (["exit", "quit", ":q"].includes(line.toLowerCase())) break;
      try {
        render(await respond(line, state, modelSpec));
      } catch (e) {
        console.error(`\n[error] ${e.message}\n`);
      }
    }
  } finally {
    rl.close();
  }
}
