// converse.mjs — the one conversation turn, shared by the CLI and Web frontends.
//
// message + session state → (LLM intent route) → (deterministic C-arm brain) → structured result.
// The LLM only routes intent and narrates; genes come solely from the Python pipeline.

import { routeIntent } from "./intent.mjs";
import { complete } from "./llm/complete.mjs";
import { suggestGenes, simulateCampaign, DATASETS } from "./brain.mjs";

function mergeUnique(existing, incoming) {
  const set = new Set(existing || []);
  for (const g of incoming || []) set.add(g);
  return [...set];
}

async function narrateBatch(dataset, genes, modelSpec) {
  try {
    const prompt =
      `For the CRISPR phenotype "${dataset}", the deterministic C-arm model recommends testing these ` +
      `${genes.length} genes next: ${genes.join(", ")}.\n` +
      `Write 1-2 short sentences on the biological theme of this batch. Do NOT add or remove genes.`;
    return await complete({ spec: modelSpec, prompt, temperature: 0.2, maxTokens: 200 });
  } catch {
    return null;
  }
}

function defaultAsk() {
  return `Which supported phenotype are you screening? Options: ${DATASETS.join(", ")}.`;
}

/**
 * @param {string} message
 * @param {{dataset:?string, hits:string[], misses:string[]}} state  mutated in place
 * @param {string} modelSpec
 * @returns {Promise<object>} one of:
 *   { kind:"chat", text, state }
 *   { kind:"simulate", text, dataset, state }
 *   { kind:"suggest", dataset, genes, info, narration, feedback:{hits,misses}, state }
 */
export async function respond(message, state, modelSpec) {
  const intent = await routeIntent(message, state, modelSpec);

  if (intent.dataset && DATASETS.includes(intent.dataset)) state.dataset = intent.dataset;
  state.hits = mergeUnique(state.hits, intent.new_hits);
  state.misses = mergeUnique(state.misses, intent.new_misses);

  if (intent.action === "chat" || !state.dataset) {
    return { kind: "chat", text: intent.reply || defaultAsk(), state };
  }

  if (intent.action === "simulate") {
    const text = await simulateCampaign({ dataset: state.dataset, rounds: intent.rounds || 5 });
    return { kind: "simulate", text, dataset: state.dataset, state };
  }

  const r = await suggestGenes({
    dataset: state.dataset,
    n: intent.n || undefined,
    testedHits: state.hits,
    testedMisses: state.misses,
  });
  const narration = await narrateBatch(state.dataset, r.genes, modelSpec);
  return {
    kind: "suggest",
    dataset: state.dataset,
    genes: r.genes,
    info: r.info,
    narration,
    feedback: { hits: state.hits.length, misses: state.misses.length },
    state,
  };
}

export { DATASETS };
