// converse.mjs — the one conversation turn, shared by the CLI and Web frontends.
//
// message + session state → (LLM intent route) → (deterministic C-arm brain) → structured result.
// The LLM only routes intent and narrates; genes come solely from the Python pipeline.

import { routeIntent } from "./intent.mjs";
import { complete } from "./llm/complete.mjs";
import { suggestGenes, simulateCampaign, getDatasets } from "./brain.mjs";
import { startCampaign, commitRound, submitResults, finish, campaignCommand } from "./campaign.mjs";
import { startOnboarding, handleOnboarding } from "./onboard.mjs";

function looksLikePath(s) {
  return /\.(csv|txt|tsv)$/i.test(s) || s.startsWith("/") || s.startsWith("~") || s.startsWith("./");
}

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

async function defaultAsk() {
  const ds = await getDatasets();
  return `Which supported phenotype are you screening? Options: ${ds.join(", ")}.`;
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
export async function respond(message, state, modelSpec, opts = {}) {
  // Onboarding the scientist's own phenotype (describe → gene pool → anchors → build).
  if (state.onboarding?.active) {
    return { ...(await handleOnboarding(state, message, opts, modelSpec)), state };
  }

  // Active experiment campaign: deterministic control (no LLM needed).
  if (state.campaign?.active) {
    const c = state.campaign;
    // Upload mode: awaiting the scientist's screen readout for the committed batch.
    if (c.stage === "awaiting_results") {
      if (campaignCommand(message) === "stop") return { ...finish(state), state };
      if (opts.file?.content != null) {
        return { ...(await submitResults(state, { content: opts.file.content, name: opts.file.name, scoreCol: opts.scoreCol, topRatio: opts.topRatio })), state };
      }
      const path = (opts.path || message || "").trim();
      if (path && looksLikePath(path)) {
        return { ...(await submitResults(state, { path, scoreCol: opts.scoreCol, topRatio: opts.topRatio })), state };
      }
      return {
        kind: "campaign", stage: "awaiting_results", dataset: c.dataset, round: c.round, genes: c.pending,
        text: "Provide this round's screen readout to reveal hits — upload a file (Web) or paste a file path (MAGeCK gene_summary or Gene,Score CSV). Or type `stop` to end.",
        state,
      };
    }
    // Proposed batch: awaiting commit.
    const cmd = campaignCommand(message);
    if (cmd === "commit") return { ...(await commitRound(state)), state };
    if (cmd === "stop") return { ...finish(state), state };
    return {
      kind: "campaign", stage: "await", dataset: c.dataset,
      text: `Type \`commit\` to test round ${c.round}'s batch, or \`stop\` to end.`,
      state,
    };
  }

  const intent = await routeIntent(message, state, modelSpec);

  const known = await getDatasets();
  if (intent.dataset && known.includes(intent.dataset)) state.dataset = intent.dataset;
  state.hits = mergeUnique(state.hits, intent.new_hits);
  state.misses = mergeUnique(state.misses, intent.new_misses);

  // The scientist's own screen: build its features before anything can be ranked.
  if (intent.action === "register") {
    const started = await startOnboarding(
      state,
      { name: intent.name, task: intent.task, measurement: intent.measurement, batchSize: intent.n },
      modelSpec,
    );
    return { ...started, state };
  }

  if (intent.action === "chat" || !state.dataset) {
    return { kind: "chat", text: intent.reply || (await defaultAsk()), state };
  }

  if (intent.action === "experiment") {
    const started = await startCampaign(state, {
      dataset: state.dataset, rounds: intent.rounds, batchSize: intent.n, mode: intent.mode,
    });
    return { ...started, state };
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

export { getDatasets };
