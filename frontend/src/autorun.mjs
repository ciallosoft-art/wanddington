// autorun.mjs — retrospective "one-click" evaluation.
//
// The real production question: a scientist uploads a COMPLETED screen (a MAGeCK gene_summary or a
// Gene,Score readout). Can the system, on its own, run the full sequential C-arm campaign against it
// and report the hit rate — showing each step as it goes? This module does exactly that:
//
//   recognize intent (LLM)  →  for each round: C-arm proposes a batch  →  reveal which hit by looking
//   them up in the uploaded readout (ingest.py, NOT a hidden benchmark oracle)  →  feed hits/misses
//   back  →  next round  →  cumulative hit@R5.
//
// It emits a structured event for every step (onStep), so the CLI can print a live timeline and the
// Web UI can stream it (SSE) the way Claude Code narrates its actions. Ground truth still comes only
// from the readout — nothing is fabricated; an unrun screen has no hit rate to report.
//
// Since we have no external dataset on hand, a benchmark screen's own readout
// (workspace/data/bda_benchmark/ground_truth_<name>.csv) stands in for "the user's new dataset".

import { join } from "node:path";
import { existsSync, mkdirSync, writeFileSync, readFileSync } from "node:fs";

import { suggestGenes, ingestResults, registerPhenotype, getDatasets, BENCHMARK_DATASETS, REPO_ROOT } from "./brain.mjs";
import { routeIntent } from "./intent.mjs";
import { proposeAnchors } from "./onboard.mjs";

// A few benchmark screens carry a different file name than their pipeline id.
const READOUT_NAME = { Steinhart: "Steinhart_crispra_GD2_D22", Carnevale22: "Carnevale22_Adenosine" };
const USER_DIR = join(REPO_ROOT, "workspace", "data", "user_phenotypes");

/**
 * Resolve a screen's readout file. For a benchmark screen it is the bundled ground-truth (demo
 * stand-in); for a scientist's REGISTERED phenotype it is the readout they uploaded, saved next to
 * the phenotype (workspace/data/user_phenotypes/<name>/readout.csv).
 */
export function readoutFor(dataset) {
  const user = join(USER_DIR, dataset, "readout.csv");
  if (existsSync(user)) return user;
  const nm = READOUT_NAME[dataset] || dataset;
  const p = join(REPO_ROOT, "workspace", "data", "bda_benchmark", `ground_truth_${nm}.csv`);
  return existsSync(p) ? p : null;
}

/**
 * Run the sequential campaign against a completed-screen readout, streaming each step.
 *
 * @param {object}   o
 * @param {string}   o.dataset       phenotype id used for ranking (suggest.py)
 * @param {string}  [o.readoutFile]  the completed-screen readout; defaults to the benchmark stand-in
 * @param {number}  [o.rounds=5]     evaluation rounds
 * @param {number}  [o.batchSize]    genes per round (default: the dataset's own batch size)
 * @param {number}  [o.topRatio=0.05] top-fraction of the readout counted as hits (ingest.py)
 * @param {string}  [o.modelSpec]    chat model for the intent step
 * @param {(e:object)=>void} [o.onStep]  receives each step event
 * @returns {Promise<object>} summary { dataset, rounds, hit_at_r5, cumulative, total, perRound, tested }
 */
export async function autoRun({ dataset, readoutFile, rounds = 5, batchSize, topRatio = 0.05, modelSpec, onStep = () => {} }) {
  const emit = (e) => { try { onStep({ t: Date.now(), ...e }); } catch { /* a dead stream must not kill the run */ } };

  const known = await getDatasets();
  if (!known.includes(dataset)) throw new Error(`unknown phenotype "${dataset}". Supported: ${known.join(", ")}`);
  const readout = readoutFile || readoutFor(dataset);
  if (!readout) throw new Error(`no readout file found for "${dataset}" (looked in workspace/data/bda_benchmark)`);

  // Step 0 — the LLM reads the request and states the plan (intent recognition).
  emit({ type: "intent-start", dataset });
  let intentText = `Run the sequential gene-selection campaign on ${dataset} for ${rounds} rounds and report hit@R5.`;
  try {
    const it = await routeIntent(
      `I uploaded my ${dataset} screen readout — run the gene selection round by round and tell me the final hit rate.`,
      { dataset }, modelSpec,
    );
    if (it?.action) intentText = `Understood: ${it.action}${it.rounds ? ` (${it.rounds} rounds)` : ""} on ${dataset}, scoring hits from your uploaded readout each round.`;
  } catch { /* narration only; fall back to the default text */ }
  emit({ type: "intent", text: intentText, dataset, rounds, readout });

  // Where the cross-experiment prior's training signal comes from. A benchmark target is held out
  // (LOO → the other 8 screens); a NEW registered screen is not in the benchmark at all, so the prior
  // trains on ALL of them — zero-shot transfer at round 1, before any in-experiment label exists.
  const isBenchmark = (BENCHMARK_DATASETS || []).includes(dataset);
  const refScreens = Math.max(0, (BENCHMARK_DATASETS || []).length - (isBenchmark ? 1 : 0));

  const testedHits = [], testedMisses = [];
  let cumulative = 0, total = null;
  const perRound = [];

  for (let round = 1; round <= rounds; round++) {
    emit({ type: "round-open", round, rounds });

    // The C-arm ranks and proposes the next batch (feedback-aware; trace records WHY).
    const sug = await suggestGenes({
      dataset, n: batchSize || undefined,
      testedHits, testedMisses, exclude: [...testedHits, ...testedMisses], trace: true,
    });
    const tr = sug.info?.trace || {};
    const sources = (tr.genes || []).reduce((m, g) => { m[g.gene] = g.source; return m; }, {});
    const inExpLabels = testedHits.length + testedMisses.length;  // revealed so far (0 at round 1)
    const transfer = { refScreens, isBenchmark, inExpLabels, zeroShot: inExpLabels === 0 };
    emit({ type: "proposed", round, genes: sug.genes, counts: tr.counts || null, shap: tr.shap || null, w_ml: tr.w_ml, w_llm: tr.w_llm, sources, transfer });

    // Reveal outcomes from the uploaded readout (the real production reveal — not a hidden oracle).
    const ing = await ingestResults({ genes: sug.genes, path: readout, topRatio });
    total = ing.totalHits;
    const nHits = ing.hits.length;
    cumulative += nHits;
    testedHits.push(...ing.hits);
    testedMisses.push(...ing.misses);
    const ratio = total ? cumulative / total : 0;
    perRound.push({ round, tested: sug.genes.length, nHits, cumulative, total, ratio });
    emit({ type: "revealed", round, hits: ing.hits, nHits, tested: sug.genes.length, cumulative, total, ratio });
  }

  const summary = {
    dataset, rounds, readout,
    hit_at_r5: perRound.at(-1)?.ratio ?? 0,
    cumulative, total, perRound,
    tested: { hits: testedHits, misses: testedMisses },
  };
  emit({ type: "summary", ...summary });
  return summary;
}

/**
 * The real production path: a scientist uploads a COMPLETED screen readout for a brand-new phenotype.
 * We register it (build its ML features so it becomes rankable — no code change, just data), save the
 * readout as its ground truth, then stream the same auto-run. The one uploaded Gene,Score file is both
 * the gene pool (for registration) and the reveal source (for scoring).
 *
 * @param {object} o
 * @param {string} o.name             a simple phenotype id (letters/digits/underscore)
 * @param {string} [o.readoutFile]    path to the completed-screen readout (CLI)
 * @param {string} [o.readoutContent] raw readout content (Web upload)
 * @param {string} [o.readoutName]    original filename (extension hint for parsing)
 * @param {string[]} [o.anchors]      seed genes; if omitted, the LLM proposes them from the task
 * @param {string} [o.task] [o.measurement] phenotype description (LLM narration + anchor proposal)
 */
export async function autoRunUpload({ name, readoutFile, readoutContent, readoutName, anchors, task, measurement, batchSize, rounds = 5, topRatio = 0.05, modelSpec, onStep = () => {} }) {
  const emit = (e) => { try { onStep({ t: Date.now(), ...e }); } catch { /* dead stream */ } };
  if (!name || /[^A-Za-z0-9_]/.test(name)) throw new Error("upload run: a simple phenotype name (letters/digits/underscore only) is required");
  const content = readoutContent != null ? readoutContent : (readoutFile ? readFileSync(readoutFile, "utf8") : null);
  if (content == null) throw new Error("upload run: need a readout file (path or content)");

  // Save the uploaded readout as this phenotype's ground truth (also lets future runs find it).
  const dir = join(USER_DIR, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "readout.csv"), content);
  emit({ type: "upload", name, bytes: content.length });

  // Register (build features) unless it is already rankable.
  const known = await getDatasets();
  if (!known.includes(name)) {
    const t = task || `CRISPR screen: ${name}`;
    const m = measurement || "screen readout score (sgRNA log-fold-change)";
    let anch = (anchors || []).map((a) => String(a).trim().toUpperCase()).filter(Boolean);
    if (!anch.length) {
      emit({ type: "anchor-proposing", task: t });
      try { anch = await proposeAnchors(t, m, modelSpec); } catch { anch = []; }
    }
    if (!anch.length) throw new Error("upload run: no anchor genes — pass anchors, or give a task the model can seed anchors from");
    emit({ type: "anchors", anchors: anch });
    emit({ type: "registering", name });
    const reg = await registerPhenotype({ name, genesContent: content, genesName: readoutName || "readout.csv", anchors: anch, task: t, measurement: m, batchSize });
    emit({ type: "registered", name, nGenes: reg.n_genes, nWithFeatures: reg.n_genes_with_known_features, anchors: anch });
  } else {
    emit({ type: "already-registered", name });
  }

  return autoRun({ dataset: name, readoutFile: join(dir, "readout.csv"), rounds, batchSize, topRatio, modelSpec, onStep });
}
