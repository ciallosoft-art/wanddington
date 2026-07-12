// brain.mjs — the gene-selection "brain" seam.
//
// The conversational shell never decides genes itself. It shells out to the deterministic Python
// C-arm pipeline (the benchmark-winning system, hit@R5 = 0.256), exactly the pattern the archived
// app used (execFile → a Python ranker) — but re-pointed from the old System-1 static ranker to the
// current `waddington_select.suggest` / `.simulate`.

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { dirname, resolve, join, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";

const execFileAsync = promisify(execFile);

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = resolve(HERE, "..", ".."); // frontend/src → frontend → waddington repo

// How to invoke the Python package. Override with WADDINGTON_PY (e.g. a bare "python3" inside the
// activated env). Default assumes conda is on PATH.
const PY_CMD = (process.env.WADDINGTON_PY || "conda run -n waddington-bio python3").split(/\s+/);

export const DATASETS = [
  "IFNG", "IL2", "Sanchez21", "Sanchez21_down", "Carnevale22",
  "Scharenberg22", "Steinhart", "Replogle_K562_essential", "Replogle_K562_gwps",
];

function run(moduleArgs, { timeoutMs = 300_000 } = {}) {
  const [cmd, ...base] = PY_CMD;
  return execFileAsync(cmd, [...base, ...moduleArgs], {
    cwd: REPO_ROOT,
    maxBuffer: 32 * 1024 * 1024,
    timeout: timeoutMs,
  });
}

/**
 * Recommend the next batch of genes (real experiment path, feedback-aware).
 * Returns { dataset, genes: string[], info: {...} }.
 */
export async function suggestGenes({ dataset, n, testedHits, testedMisses, exclude }) {
  if (!DATASETS.includes(dataset)) {
    throw new Error(`unknown phenotype "${dataset}". Supported: ${DATASETS.join(", ")}`);
  }
  const args = ["-m", "waddington_select.suggest", "--dataset", dataset, "--json"];
  if (n) args.push("--n", String(n));
  if (testedHits?.length) args.push("--tested-hits", ...testedHits);
  if (testedMisses?.length) args.push("--tested-misses", ...testedMisses);
  if (exclude?.length) args.push("--exclude", ...exclude);

  const { stdout } = await run(args);
  const line = stdout.trim().split("\n").filter(Boolean).pop();
  if (!line) throw new Error("suggest produced no output");
  return JSON.parse(line);
}

/**
 * "Run the experiment": reveal hit/no-hit for a committed batch (the hidden oracle plays the wet lab).
 * Returns { hits: string[], reveal: {gene:bool}, nHits, totalHits }.
 */
export async function revealBatch({ dataset, genes }) {
  if (!DATASETS.includes(dataset)) {
    throw new Error(`unknown phenotype "${dataset}". Supported: ${DATASETS.join(", ")}`);
  }
  if (!genes?.length) throw new Error("revealBatch: empty batch");
  const { stdout } = await run(["-m", "waddington_select.tools", "reveal", "--dataset", dataset, "--genes", ...genes]);
  const s = stdout.indexOf("{");
  const e = stdout.lastIndexOf("}");
  if (s === -1 || e === -1) throw new Error("reveal produced no JSON");
  const r = JSON.parse(stdout.slice(s, e + 1));
  return { hits: r.hits || [], reveal: r.reveal || {}, nHits: r.n_hits ?? (r.hits?.length || 0), totalHits: r.total_hits };
}

/**
 * "Real experiment" reveal: derive hits for a tested batch from the scientist's uploaded screen
 * readout (MAGeCK gene_summary / Gene,Score CSV / labelled table). Pass either a file `path` or raw
 * `content` (+ `name` for the extension hint). Returns { hits, misses, totalHits }.
 */
export async function ingestResults({ genes, path, content, name, scoreCol, topRatio }) {
  if (!genes?.length) throw new Error("ingestResults: empty batch");
  let filePath = path;
  if (!filePath) {
    if (content == null) throw new Error("ingestResults: need a file path or content");
    const ext = (name && extname(name)) || ".txt"; // .csv → comma, else tab (per ingest.py)
    const dir = mkdtempSync(join(tmpdir(), "wadd-results-"));
    filePath = join(dir, `upload${ext}`);
    writeFileSync(filePath, content);
  }
  const args = ["-m", "waddington_select.ingest", "--file", filePath, "--genes", ...genes, "--json"];
  if (scoreCol) args.push("--score-col", scoreCol);
  if (topRatio) args.push("--top-ratio", String(topRatio));
  const { stdout } = await run(args);
  const s = stdout.indexOf("{");
  const e = stdout.lastIndexOf("}");
  if (s === -1 || e === -1) throw new Error(`ingest failed: ${stdout.slice(0, 200)}`);
  const r = JSON.parse(stdout.slice(s, e + 1));
  return { hits: r.hits || [], misses: r.misses || [], totalHits: r.total_hits, unknown: r.unknown_genes || [], method: r.method };
}

/**
 * Run a narrated oracle-driven demo campaign (for "show me how it would go"). Returns raw text.
 */
export async function simulateCampaign({ dataset, rounds = 5 }) {
  if (!DATASETS.includes(dataset)) {
    throw new Error(`unknown phenotype "${dataset}". Supported: ${DATASETS.join(", ")}`);
  }
  const { stdout } = await run(
    ["-m", "waddington_select.simulate", "--dataset", dataset, "--rounds", String(rounds)],
  );
  return stdout;
}
