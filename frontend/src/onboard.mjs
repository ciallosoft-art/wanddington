// onboard.mjs — guided registration of a scientist's OWN phenotype.
//
// describe the screen → provide the gene pool file → the LLM PROPOSES anchor genes → the scientist
// confirms/edits → build features (waddington_select/phenotype.py) → it becomes rankable.
//
// Anchors are the one scientifically load-bearing input: the three anchor-relative features
// (g1_ppi_score / archs4_coexpr / kegg_overlap) are computed against them. So the LLM proposes and
// the scientist vets. They are seeds for feature computation only — suggest.py never recommends a
// phenotype's own anchors back, so a mediocre anchor cannot resurface as a fake "discovery".

import { complete } from "./llm/complete.mjs";
import { registerPhenotype } from "./brain.mjs";

const EXTRACT_SYS = `Extract a CRISPR screen registration from the scientist's message.
Return ONE JSON object, nothing else:
{
  "name": <short identifier, no spaces (e.g. "Ferroptosis_K562"); null if not inferable>,
  "task": <one sentence: what phenotype they are screening for; null if unclear>,
  "measurement": <what the screen readout measures (e.g. "log-fold-change of sgRNA abundance"); null if unclear>,
  "batch_size": <genes they can test per round, or null>
}
Infer a sensible name from the biology if they didn't give one. Do not invent a task/measurement that
they did not describe — use null so we can ask.`;

const ANCHOR_SYS = `You are proposing ANCHOR genes that seed a gene-selection model for a new CRISPR screen.

Anchors are 8-12 well-established human genes CENTRAL to the phenotype's biology (canonical pathway
members, receptors, core regulators). They are used only to compute network / co-expression / pathway
similarity features for every gene in the library. They are NOT predictions and are never recommended
back to the scientist, so choose genes that are textbook-certain for this biology rather than guesses.

Return ONLY a JSON array of official human gene symbols, e.g. ["GPX4","SLC7A11","ACSL4"]`;

const GENE_RE = /\b[A-Z][A-Z0-9-]{1,14}\b/g;
const STOP = new Set([
  "YES", "NO", "OK", "OKAY", "USE", "THE", "AND", "OR", "GENES", "GENE", "ANCHORS", "ANCHOR",
  "CONFIRM", "INSTEAD", "ADD", "REMOVE", "DROP", "PLEASE", "CSV", "TSV", "TXT", "MAGECK", "I", "A",
]);

function parseJson(text) {
  let t = String(text).trim();
  if (t.startsWith("```")) t = t.replace(/^```[a-zA-Z]*\n?/, "").replace(/\n?```$/, "").trim();
  try {
    return JSON.parse(t);
  } catch {
    const m = t.match(/[[{][\s\S]*[\]}]/);
    if (m) {
      try {
        return JSON.parse(m[0]);
      } catch {
        /* fall through */
      }
    }
  }
  return null;
}

/** Gene symbols the scientist typed (used to override the proposed anchors). */
export function parseGenes(message) {
  const hits = String(message).match(GENE_RE) || [];
  return [...new Set(hits.filter((g) => !STOP.has(g) && /\d|[A-Z]{2,}/.test(g)))];
}

export function onboardCommand(message) {
  const m = String(message).trim().toLowerCase();
  if (/^(yes|y|ok|okay|confirm|looks good|accept|go|use them|proceed)\b/.test(m) || /确认|可以|好的|同意|就用/.test(message)) return "confirm";
  if (/\b(cancel|abort|stop|quit|nevermind)\b/.test(m) || /取消|算了|停止/.test(message)) return "cancel";
  return "other";
}

export async function extractPhenotype(message, modelSpec) {
  try {
    const raw = await complete({ spec: modelSpec, systemPrompt: EXTRACT_SYS, prompt: message, temperature: 0, maxTokens: 400 });
    return parseJson(raw) || {};
  } catch {
    return {};
  }
}

export async function proposeAnchors(task, measurement, modelSpec) {
  const raw = await complete({
    spec: modelSpec,
    systemPrompt: ANCHOR_SYS,
    prompt: `Phenotype: ${task}\nMeasurement: ${measurement}\n\nPropose the anchor genes.`,
    temperature: 0,
    maxTokens: 300,
  });
  const arr = parseJson(raw);
  const genes = Array.isArray(arr) ? arr : parseGenes(raw);
  return [...new Set(genes.map((g) => String(g).trim().toUpperCase()).filter((g) => GENE_RE.test(g)))].slice(0, 12);
}

/** Begin onboarding; immediately advances to the first thing we actually need. */
export async function startOnboarding(state, fields, modelSpec) {
  state.onboarding = {
    active: true,
    stage: "collect",
    name: fields.name || null,
    task: fields.task || null,
    measurement: fields.measurement || null,
    batchSize: fields.batchSize || null,
    genesPath: null,
    genesContent: null,
    genesName: null,
    proposed: null,
    anchors: null,
  };
  return advance(state, modelSpec);
}

/** Ask for whatever is still missing; when everything is in, build the phenotype. */
async function advance(state, modelSpec) {
  const o = state.onboarding;

  if (!o.name || !o.task || !o.measurement) {
    o.stage = "collect";
    const missing = [
      !o.name && "a short name for the screen",
      !o.task && "what phenotype you're screening for",
      !o.measurement && "what your readout measures",
    ].filter(Boolean);
    return {
      kind: "onboard", stage: "collect",
      text: `Let's onboard your screen. I still need: ${missing.join("; ")}.`,
      have: { name: o.name, task: o.task, measurement: o.measurement },
    };
  }

  if (!o.genesPath && !o.genesContent) {
    o.stage = "pool";
    return {
      kind: "onboard", stage: "pool", name: o.name,
      text: `Now provide your screen/library file — every gene in it becomes the candidate pool (MAGeCK gene_summary or a Gene,Score CSV).`,
    };
  }

  if (!o.anchors) {
    o.stage = "anchors";
    if (!o.proposed) o.proposed = await proposeAnchors(o.task, o.measurement, modelSpec);
    return {
      kind: "onboard", stage: "anchors", name: o.name, proposed: o.proposed,
      text: `Proposed anchor genes (seeds used only to compute similarity features — they'll never be recommended back to you). Reply \`yes\` to accept, or list your own gene symbols instead.`,
    };
  }

  o.stage = "building";
  const res = await registerPhenotype({
    name: o.name,
    genesFile: o.genesPath || undefined,
    genesContent: o.genesContent || undefined,
    genesName: o.genesName || undefined,
    anchors: o.anchors,
    task: o.task,
    measurement: o.measurement,
    batchSize: o.batchSize || undefined,
  });
  state.onboarding = null;
  state.dataset = res.name;
  return { kind: "onboard", stage: "done", ...res };
}

/** One turn while onboarding is active. */
export async function handleOnboarding(state, message, opts, modelSpec) {
  const o = state.onboarding;
  const cmd = onboardCommand(message);
  if (cmd === "cancel") {
    state.onboarding = null;
    return { kind: "onboard", stage: "cancelled", text: "Onboarding cancelled." };
  }

  if (o.stage === "collect") {
    const f = await extractPhenotype(message, modelSpec);
    o.name = o.name || f.name || null;
    o.task = o.task || f.task || null;
    o.measurement = o.measurement || f.measurement || null;
    o.batchSize = o.batchSize || (Number.isInteger(f.batch_size) ? f.batch_size : null);
    return advance(state, modelSpec);
  }

  if (o.stage === "pool") {
    if (opts.file?.content != null) {
      o.genesContent = opts.file.content;
      o.genesName = opts.file.name;
    } else {
      const p = (opts.path || message || "").trim();
      if (p && /\.(csv|txt|tsv)$/i.test(p)) o.genesPath = p;
    }
    return advance(state, modelSpec);
  }

  if (o.stage === "anchors") {
    if (cmd === "confirm") {
      o.anchors = o.proposed;
    } else {
      const custom = parseGenes(message);
      if (custom.length >= 3) o.anchors = custom;
    }
    if (!o.anchors) {
      return {
        kind: "onboard", stage: "anchors", name: o.name, proposed: o.proposed,
        text: "Reply `yes` to accept these anchors, or list at least 3 gene symbols to use instead.",
      };
    }
    return advance(state, modelSpec);
  }

  return advance(state, modelSpec);
}
