// chat.mjs — the tool-less terminal conversation (thin renderer over converse.respond).

import { createInterface } from "node:readline/promises";
import { stdin, stdout } from "node:process";

import { DEFAULT_CHAT_MODEL } from "./intent.mjs";
import { respond, DATASETS } from "./converse.mjs";

const BANNER = `Waddington — conversational gene selection (tool-less, C-arm pipeline).
Tell me a phenotype and I'll recommend genes to perturb; report results and I'll adapt.
Supported phenotypes: ${DATASETS.join(", ")}.
Type "exit" to quit.\n`;

function render(result) {
  if (result.kind === "chat") {
    console.log(`\n${result.text}\n`);
    return;
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
  console.log(BANNER);
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
