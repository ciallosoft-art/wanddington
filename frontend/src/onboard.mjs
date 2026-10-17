// onboard.mjs — guided registration of a scientist's OWN phenotype.
//
// describe the screen → provide the gene pool file → tier → hit rule → the screen's design (perturbation,
// cell line, curated panel?) → build features (waddington_select/phenotype.py) → it becomes rankable.
//
// The design decides which DepMap essentiality features the prior may use, by the benchmarks' metadata
// rule (router_protocol.feature_policy_from): none for CRISPRa (DepMap is knockout-derived) or a curated
// essential panel, pan-cancer only in K562, else pan-cancer + K562. Anchor genes are no longer asked for:
// the reported C-arm drops the anchor-relative features, so they would not change a single ranking.

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

// Whole-token validator. Deliberately NOT global: a /g regex carries lastIndex across .test()
// calls, which silently drops every other gene when used in a filter. Anchored too, so a token
// only passes if the WHOLE string is a symbol -- \b alone accepts "TP53 (tumour suppressor)".
const GENE_TOKEN_RE = /^[A-Z][A-Z0-9-]{1,14}$/;

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

export function onboardCommand(message) {
  const m = String(message).trim().toLowerCase();
  if (/^(yes|y|ok|okay|confirm|looks good|accept|go|use them|proceed)\b/.test(m) || /确认|可以|好的|同意|就用/.test(message)) return "confirm";
  if (/\b(cancel|abort|stop|quit|nevermind)\b/.test(m) || /取消|算了|停止/.test(message)) return "cancel";
  return "other";
}

/**
 * The single-gene tier a reply asks for: "non_essential" (remove the 684 CEGv2 core-essential genes — the
 * default and the paper's main tier) or "full_pool" (keep them), else null. A bare "yes"/"ok" is handled by
 * the caller as accepting the default.
 */
export function parseTier(message) {
  const m = String(message || "").trim().toLowerCase();
  if (/\b(keep|full|all genes|include|no)\b/.test(m) || /全池|保留|不排除|不去掉|全部基因/.test(message)) return "full_pool";
  if (/\b(remove|exclude|drop|default|non[-_ ]?essential)\b/.test(m) || /排除|去掉|去除|默认|非必需/.test(message)) return "non_essential";
  return null;
}

/**
 * The hit rule a reply describes (ingest.py): {type:"label"} | {type:"threshold", threshold, direction} |
 * {type:"fdr", maxFdr, direction} | {type:"top_fraction", fraction, direction}, else null. Directions:
 * "both" (|score|), "up" or "down". The caller refuses top_fraction for a campaign uploaded batch by batch.
 */
export function parseHitRule(message) {
  const m = String(message || "").trim().toLowerCase().replace(/≥/g, ">=").replace(/≤/g, "<=");
  const dir = /\b(up|pos|positive|increase[sd]?)\b|上调|升高|增加/.test(m) ? "up"
    : /\b(down|neg|negative|decrease[sd]?)\b|下调|降低|减少/.test(m) ? "down" : "both";
  const num = (s) => Number(s);
  let x;
  if ((x = m.match(/\b(?:fdr|padj|q-?value|q)\s*(?:<=|<|小于)\s*(\d*\.?\d+)/))) {
    const q = num(x[1]);
    return q > 0 && q <= 1 ? { type: "fdr", maxFdr: q, direction: dir } : null;
  }
  if ((x = m.match(/(?:\btop|前)\s*(\d*\.?\d+)\s*%/))) {
    const f = num(x[1]) / 100;
    return f > 0 && f <= 1 ? { type: "top_fraction", fraction: f, direction: dir } : null;
  }
  const pos = (v, d) => (v > 0 ? { type: "threshold", threshold: v, direction: d } : null);
  if ((x = m.match(/\|\s*[a-z_ -]*\s*\|\s*(?:>=|>)\s*(\d*\.?\d+)/))) return pos(num(x[1]), "both");
  if ((x = m.match(/\babs(?:olute)?\b[^0-9]*?(\d*\.?\d+)/))) return pos(num(x[1]), "both");
  if ((x = m.match(/\b(?:score|lfc|logfc|z)\s*(?:<=|<)\s*-\s*(\d*\.?\d+)/))) return pos(num(x[1]), "down");
  if ((x = m.match(/\b(?:score|lfc|logfc|z)\s*(?:>=|>)\s*(\d*\.?\d+)/))) return pos(num(x[1]), "up");
  if (/\b(label|labels|labell?ed|hit column)\b|标签|自己标|hit ?列/.test(m)) return { type: "label" };
  return null;
}

const HITS_TEXT =
  "How should a tested gene count as a hit? This stays fixed for the whole campaign, so decide it before you see " +
  "any results. Reply with one of: `label` — your result files will carry a hit column (1/0, yes/no); " +
  "`|score| >= 1.5` — an absolute cutoff on the score (or `score >= 1.5` / `score <= -1.5` for one direction); " +
  "`fdr < 0.05` — an FDR cutoff, e.g. MAGeCK's (add `up` or `down` for one direction).";

const TOP_FRACTION_REFUSED =
  "A `top 5%` rule ranks each gene against the whole screen, but you will upload one round's batch at a time, so it " +
  "would call ~5% of every batch a hit whatever the effects. Choose `label`, a score cutoff (`|score| >= 1.5`) or an " +
  "FDR cutoff (`fdr < 0.05`). (To replay a completed screen, use upload-and-run instead, which can rank the whole screen.)";

// The perturbation, in the order checked: CRISPRa and CRISPRi before KO, so "dCas9-KRAB" is not a knockout.
// A bare "activation" is not CRISPRa: "T cell activation" is a phenotype, not a modality.
const MODALITY_RES = [
  ["CRISPRa", /crispr-?a\b|crispr[- ]activation|vp64|\bvpr\b|\bsam\b|over-?express|gain[- ]of[- ]function|crispr激活|过表达/i],
  ["CRISPRi", /crispr-?i\b|crispr[- ]interference|knock-?down|krab|crispr干扰|敲低/i],
  ["KO", /\bko\b|knock-?out|crispr-?ko|\bcas9\b|敲除/i],
];
const UNKNOWN_RE = /\b(?:unknown|unsure|not sure|don'?t know)\b|不知道|不确定|未知/i;
const K562_RE = /(?<![A-Za-z0-9])K-?562(?![0-9])/i;
const CURATED_RE = /\bcurated\b|essential[- ](?:gene )?panel|必需基因(?:面板|集合|库)/i;
const NOT_CURATED_RE = /\b(?:not|no|isn'?t|non)[- ](?:a\s+)?curated|不是/i;

/** KO / CRISPRi / CRISPRa named in a message, "unknown" if they say they don't know, else null. */
export function parseModality(message) {
  const m = String(message || "");
  for (const [modality, re] of MODALITY_RES) if (re.test(m)) return modality;
  return UNKNOWN_RE.test(m) ? "unknown" : null;
}

/**
 * The screen's design from a reply such as "CRISPRa, primary T cells" or "KO in K562":
 * {modality: KO|CRISPRi|CRISPRa|unknown|null, cellLine: string|null, curatedEss: boolean}.
 */
export function parseDesign(message) {
  const text = String(message || "").trim();
  const curatedEss = CURATED_RE.test(text) && !NOT_CURATED_RE.test(text);
  let cellLine = null;
  if (K562_RE.test(text)) cellLine = "K562";
  else {
    const inPhrase = text.match(/\bin\s+([^,;.，；。]+)/i) || text.match(/在\s*([^,;.，；。中里]+)[中里]/);
    const part = text.split(/[,;，；\n]/).map((x) => x.trim()).filter(Boolean)
      .find((x) => !parseModality(x) && !CURATED_RE.test(x) && onboardCommand(x) === "other");
    cellLine = (inPhrase ? inPhrase[1] : part || "").trim().slice(0, 80) || null;
  }
  return { modality: parseModality(text), cellLine, curatedEss };
}

/** What a description already says about the design (the scientist confirms it): a named perturbation, else null. */
function designGuess(description) {
  const modality = parseModality(description);
  if (!modality || modality === "unknown") return null;
  return { modality, cellLine: K562_RE.test(description) ? "K562" : null,
           curatedEss: CURATED_RE.test(description) && !NOT_CURATED_RE.test(description) };
}

const DESIGN_WHY = "It decides which DepMap essentiality features the model may use (none for CRISPRa or a curated " +
  "essential panel, pan-cancer only in K562).";

const DESIGN_TEXT =
  "Last, the screen's design. Which perturbation — `KO`, `CRISPRi` or `CRISPRa` — and which cells? e.g. " +
  "`CRISPRa, primary T cells` or `CRISPRi, K562`. Add `curated essential panel` if your library is a curated set of " +
  "essential genes; say `unknown` if you don't know the perturbation (it is then treated as loss-of-function). " + DESIGN_WHY;

export function designText(guess) {
  if (!guess) return DESIGN_TEXT;
  const where = guess.cellLine ? ` in ${guess.cellLine}` : "";
  return `I read the design as ${guess.modality}${where}${guess.curatedEss ? ", a curated essential panel" : ""}. ` +
    "Reply `yes` if that's right, or correct it (e.g. `CRISPRi, K562`; add the cells if you like). " + DESIGN_WHY;
}

const TIER_TEXT =
  "Should the 684 common core-essential genes (CEGv2) be removed from the candidates? Reply `yes` to remove them " +
  "(the default, and the paper's main tier: you are after phenotype-specific hits, not generic essentials), or " +
  "`keep` to rank the full pool, essentials included.";

export async function extractPhenotype(message, modelSpec) {
  try {
    const raw = await complete({ spec: modelSpec, systemPrompt: EXTRACT_SYS, prompt: message, temperature: 0, maxTokens: 400 });
    return parseJson(raw) || {};
  } catch {
    return {};
  }
}

/** Uppercase, de-duplicate and keep only well-formed gene symbols; at most 12. */
export function normalizeAnchors(genes) {
  return [...new Set(
    (Array.isArray(genes) ? genes : [])
      .map((g) => String(g).trim().toUpperCase())
      .filter((g) => GENE_TOKEN_RE.test(g)),
  )].slice(0, 12);
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
    description: fields.text || "",   // what the scientist has said about the screen (prefills the design)
    tier: null,
    hitRule: null,
    design: null,                     // {modality, cellLine, curatedEss}
    designGuess: null,
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

  if (!o.tier) {
    o.stage = "tier";
    return { kind: "onboard", stage: "tier", name: o.name, text: TIER_TEXT };
  }

  if (!o.hitRule) {
    o.stage = "hits";
    return { kind: "onboard", stage: "hits", name: o.name, text: HITS_TEXT };
  }

  if (!o.design) {
    o.stage = "design";
    o.designGuess = designGuess([o.description, o.task, o.measurement].filter(Boolean).join(" \n "));
    return { kind: "onboard", stage: "design", name: o.name, guess: o.designGuess, text: designText(o.designGuess) };
  }

  o.stage = "building";
  const res = await registerPhenotype({
    name: o.name,
    genesFile: o.genesPath || undefined,
    genesContent: o.genesContent || undefined,
    genesName: o.genesName || undefined,
    task: o.task,
    measurement: o.measurement,
    batchSize: o.batchSize || undefined,
    tier: o.tier,
    hitRule: o.hitRule,
    modality: o.design.modality,
    cellLine: o.design.cellLine || undefined,
    curatedEss: o.design.curatedEss,
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
    o.description = [o.description, message].filter(Boolean).join(" \n ");
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

  if (o.stage === "tier") {
    o.tier = parseTier(message) || (cmd === "confirm" ? "non_essential" : null);
    if (!o.tier) return { kind: "onboard", stage: "tier", name: o.name, text: TIER_TEXT };
    return advance(state, modelSpec);
  }

  if (o.stage === "hits") {
    const rule = parseHitRule(message);
    if (rule?.type === "top_fraction") return { kind: "onboard", stage: "hits", name: o.name, text: TOP_FRACTION_REFUSED };
    if (!rule) return { kind: "onboard", stage: "hits", name: o.name, text: HITS_TEXT };
    o.hitRule = rule;
    return advance(state, modelSpec);
  }

  if (o.stage === "design") {
    const said = parseDesign(message);
    if (said.modality) {
      o.design = said;
    } else if (cmd === "confirm" && o.designGuess) {
      // "yes" to the guess, possibly with the cells added ("yes, in K562")
      o.design = { ...o.designGuess, cellLine: said.cellLine || o.designGuess.cellLine,
                   curatedEss: said.curatedEss || o.designGuess.curatedEss };
    }
    if (!o.design) return { kind: "onboard", stage: "design", name: o.name, guess: o.designGuess, text: DESIGN_TEXT };
    return advance(state, modelSpec);
  }

  return advance(state, modelSpec);
}
