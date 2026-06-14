import { execFile } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { dbListExperiments } from "./db.js";
import { findMatchingSkill } from "./skills.js";
import { queryNeighbourhood } from "./knowledge-graph.js";

const execFileAsync = promisify(execFile);

// dist/web/gene-ranker.js → ../../ → project root → workspace/evaluation/gene_ranker.py
const _moduleDir = dirname(fileURLToPath(import.meta.url));
const RANKER_SCRIPT = resolve(_moduleDir, "..", "..", "workspace", "evaluation", "gene_ranker.py");

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface RankRequest {
  dataset?: string;        // filter context by dataset name (e.g. "IFNG")
  exclude_genes?: string[]; // genes already tested or blacklisted
  n?: number;              // number of candidates to return (default 50)
}

export interface RankedGene {
  gene: string;
  score: number;
  signals: string[];       // e.g. ["ppi_anchor", "skill_history", "kg_evidence"]
}

interface PythonRanked {
  gene: string;
  score: number;
  signals: string[];
}

// ---------------------------------------------------------------------------
// DB-derived signals (fast, no network)
// ---------------------------------------------------------------------------

function getTrueNegativeBlacklist(): Set<string> {
  const exps = dbListExperiments();
  const blacklist = new Set<string>();
  for (const exp of exps) {
    if (!exp.gene || !exp.filter_verdict_json) continue;
    try {
      const v = JSON.parse(exp.filter_verdict_json) as { verdict: string };
      if (v.verdict === "true_negative") blacklist.add(exp.gene.toUpperCase());
    } catch { /* skip */ }
  }
  return blacklist;
}

function getSkillBoosts(): Map<string, number> {
  // Genes with SKILL entries get a +0.1 score boost if success_rate >= 0.7
  const boosts = new Map<string, number>();
  const exps = dbListExperiments();
  const genes = new Set(exps.map(e => e.gene?.toUpperCase()).filter(Boolean) as string[]);
  for (const gene of genes) {
    try {
      const s = findMatchingSkill(gene);
      if (s && s.success_rate >= 0.7) boosts.set(gene, 0.1);
    } catch { /* skill lookup optional */ }
  }
  return boosts;
}

function getKgBoostedGenes(dataset?: string): Set<string> {
  // Genes with perturbed_in KG edges for this dataset context
  const boosted = new Set<string>();
  if (!dataset) return boosted;
  try {
    const { edges, nodes } = queryNeighbourhood(dataset.toUpperCase(), 2);
    const nodeMap = new Map(nodes.map(n => [n.id, n.label]));
    for (const edge of edges) {
      if (edge.rel !== "perturbed_in") continue;
      const srcLabel = nodeMap.get(edge.src) ?? edge.src;
      const dstLabel = nodeMap.get(edge.dst) ?? edge.dst;
      const dsUpper = dataset.toUpperCase();
      if (dstLabel.toUpperCase().includes(dsUpper) || srcLabel.toUpperCase().includes(dsUpper)) {
        // src is likely a gene node
        boosted.add(srcLabel.toUpperCase());
      }
    }
  } catch { /* KG optional */ }
  return boosted;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export async function rankGenes(req: RankRequest): Promise<RankedGene[]> {
  const n = req.n ?? 50;
  const dataset = req.dataset?.trim() || "";
  const excludeSet = new Set((req.exclude_genes ?? []).map(g => g.toUpperCase()));

  // 1. DB blacklist (true_negative) — add to excludes
  const blacklist = getTrueNegativeBlacklist();
  for (const g of blacklist) excludeSet.add(g);

  // 2. DB boosts
  const skillBoosts = getSkillBoosts();
  const kgBoosted   = getKgBoostedGenes(dataset);

  // 3. Call Python gene_ranker for PPI-based biological ranking
  let pythonRanked: PythonRanked[] = [];
  try {
    const excludeArg = [...excludeSet].join(",");
    const { stdout } = await execFileAsync(
      "python3",
      [RANKER_SCRIPT, dataset, "--json", "--n", String(n * 3), "--exclude", excludeArg],
      { timeout: 30_000 },
    );
    pythonRanked = JSON.parse(stdout) as PythonRanked[];
  } catch (err) {
    // Python ranker unavailable — return empty rather than crash
    console.error("[gene-ranker] Python ranker failed:", (err as Error).message);
    return [];
  }

  // 4. Merge signals and apply DB boosts
  const result: RankedGene[] = pythonRanked
    .filter(r => !excludeSet.has(r.gene.toUpperCase()))
    .map(r => {
      const gene = r.gene.toUpperCase();
      let score = r.score;
      const signals = [...r.signals];

      if (skillBoosts.has(gene)) { score += 0.1; signals.push("skill_history"); }
      if (kgBoosted.has(gene))   { score += 0.1; signals.push("kg_evidence"); }

      return { gene, score, signals };
    });

  // Sort by score descending and cap at n
  result.sort((a, b) => b.score - a.score);
  return result.slice(0, n);
}
