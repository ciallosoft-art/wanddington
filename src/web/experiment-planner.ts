import { rankGenes, type RankedGene } from "./gene-ranker.js";
import { dbListExperiments, dbListHypotheses } from "./db.js";
import { findMatchingSkill } from "./skills.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PlanRequest {
  dataset?: string;
  total_budget: number;    // total genes to test across all rounds
  batch_size?: number;     // genes per round (default: auto from rounds)
  rounds?: number;         // number of rounds (default: auto from batch_size)
  explore_ratio?: number;  // fraction of each round reserved for retries (default: 0.2)
}

export interface RoundPlan {
  round: number;
  genes: string[];
  rationale: string;
  expected_rfs_floor: number;  // conservative estimate based on SKILL history
  fallback_genes: string[];    // substitutes if a gene's environment fails
}

export interface PlanResult {
  dataset?: string;
  total_budget: number;
  batch_size: number;
  rounds: RoundPlan[];
  summary: string;
}

// ---------------------------------------------------------------------------
// DB helpers
// ---------------------------------------------------------------------------

function getNeedsInvestigationGenes(): Set<string> {
  const exps = dbListExperiments();
  const result = new Set<string>();
  for (const exp of exps) {
    if (!exp.gene || !exp.filter_verdict_json) continue;
    try {
      const v = JSON.parse(exp.filter_verdict_json) as { verdict: string };
      if (v.verdict === "needs_investigation") result.add(exp.gene.toUpperCase());
    } catch { /* skip */ }
  }
  return result;
}

function getSkillInfo(gene: string): { success_rate: number } | null {
  try {
    const s = findMatchingSkill(gene.toUpperCase());
    return s ? { success_rate: s.success_rate } : null;
  } catch { return null; }
}

function getSpeculativeHypothesisGenes(): Set<string> {
  const hyps = dbListHypotheses();
  const result = new Set<string>();
  for (const h of hyps) {
    if (h.confidence === "speculative") result.add(h.gene.toUpperCase());
  }
  return result;
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

interface ScoredCandidate {
  gene: string;
  score: number;        // final priority score
  g1_score: number;     // original G1 rank score
  signals: string[];
  skill_rate: number | null;
  is_retry: boolean;    // from needs_investigation
}

function scoreCandidate(
  ranked: RankedGene,
  retryGenes: Set<string>,
  hypothesisGenes: Set<string>,
): ScoredCandidate {
  const gene = ranked.gene.toUpperCase();
  const signals = [...ranked.signals];
  let score = ranked.score;
  let skill_rate: number | null = null;

  // SKILL boost: if we already know how to run this gene, prioritize it
  const skill = getSkillInfo(gene);
  if (skill && skill.success_rate >= 0.6) {
    score += 0.15;
    skill_rate = skill.success_rate;
    signals.push(`skill(${(skill.success_rate * 100).toFixed(0)}%)`);
  } else if (skill) {
    skill_rate = skill.success_rate;
    signals.push(`skill(${(skill.success_rate * 100).toFixed(0)}%,low)`);
  }

  // Hypothesis-driven boost: testing this gene will update a speculative hypothesis
  if (hypothesisGenes.has(gene)) {
    score += 0.10;
    signals.push("speculative_hypothesis");
  }

  const is_retry = retryGenes.has(gene);

  return { gene, score, g1_score: ranked.score, signals, skill_rate, is_retry };
}

// ---------------------------------------------------------------------------
// Round allocation
// ---------------------------------------------------------------------------

function computeExpectedRfsFloor(genes: string[]): number {
  const rates: number[] = [];
  for (const gene of genes) {
    const s = getSkillInfo(gene);
    if (s) rates.push(s.success_rate);
  }
  if (rates.length === 0) return 0.0;
  const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
  return Math.round(mean * 0.8 * 100) / 100;  // conservative: 80% of SKILL rate
}

function buildRationale(
  roundNum: number,
  skillGenes: string[],
  hypothesisGenes: string[],
  retryGenes: string[],
  plainGenes: string[],
  floor: number,
): string {
  const parts: string[] = [];
  if (skillGenes.length) {
    const avgRate = skillGenes
      .map(g => getSkillInfo(g)?.success_rate ?? 0)
      .reduce((a, b) => a + b, 0) / skillGenes.length;
    parts.push(`${skillGenes.length} SKILL-ready (avg success=${(avgRate * 100).toFixed(0)}%)`);
  }
  if (hypothesisGenes.length) parts.push(`${hypothesisGenes.length} hypothesis-driven`);
  if (retryGenes.length)      parts.push(`${retryGenes.length} needs_investigation retry`);
  if (plainGenes.length)      parts.push(`${plainGenes.length} biological-anchor candidates`);
  if (floor > 0) parts.push(`expected RFS floor ≥ ${floor}`);
  return `Round ${roundNum}: ` + (parts.join(" + ") || "random fallback");
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export async function planExperiments(req: PlanRequest): Promise<PlanResult> {
  const { dataset, explore_ratio = 0.2 } = req;

  // Resolve batch_size and rounds
  let { total_budget, batch_size, rounds } = req;
  if (!batch_size && !rounds) { batch_size = 50; }
  if (batch_size && !rounds) { rounds = Math.ceil(total_budget / batch_size); }
  if (rounds && !batch_size) { batch_size = Math.ceil(total_budget / rounds); }
  batch_size = batch_size!;
  rounds = rounds!;

  // 1. Get ranked candidates from G1 (fetch 3× budget so we have fallbacks)
  const poolSize = Math.max(total_budget * 3, 200);
  const rankedRaw = await rankGenes({ dataset, n: poolSize });

  // 2. DB-derived signals
  const retryGenes      = getNeedsInvestigationGenes();
  const hypothesisGenes = getSpeculativeHypothesisGenes();

  // 3. Score all candidates
  const scored: ScoredCandidate[] = rankedRaw.map(r =>
    scoreCandidate(r, retryGenes, hypothesisGenes),
  );

  // Separate retry pool from main pool; sort each by score descending
  const retryPool  = scored.filter(c => c.is_retry).sort((a, b) => b.score - a.score);
  const mainPool   = scored.filter(c => !c.is_retry).sort((a, b) => b.score - a.score);

  // 4. Allocate to rounds
  const assignedGenes = new Set<string>();
  const roundPlans: RoundPlan[] = [];
  let retryIdx = 0, mainIdx = 0;

  for (let r = 1; r <= rounds; r++) {
    const retryBudget = Math.floor(batch_size * explore_ratio);
    const mainBudget  = batch_size - retryBudget;

    const roundRetry: ScoredCandidate[] = [];
    const roundMain:  ScoredCandidate[] = [];

    // Fill retry slot
    while (roundRetry.length < retryBudget && retryIdx < retryPool.length) {
      const c = retryPool[retryIdx++]!;
      if (!assignedGenes.has(c.gene)) { roundRetry.push(c); assignedGenes.add(c.gene); }
    }

    // Fill main slots (skill-first, then hypothesis, then plain)
    const candidatesForRound = mainPool
      .filter(c => !assignedGenes.has(c.gene))
      .slice(0, mainBudget * 3);  // take 3× and pick by bucket order

    const skillCandidates      = candidatesForRound.filter(c => c.skill_rate !== null && c.skill_rate >= 0.6);
    const hypothesisCandidates = candidatesForRound.filter(c => c.signals.includes("speculative_hypothesis") && !skillCandidates.some(s => s.gene === c.gene));
    const plainCandidates      = candidatesForRound.filter(c => !skillCandidates.some(s => s.gene === c.gene) && !hypothesisCandidates.some(h => h.gene === c.gene));

    for (const bucket of [skillCandidates, hypothesisCandidates, plainCandidates]) {
      for (const c of bucket) {
        if (roundMain.length >= mainBudget) break;
        if (!assignedGenes.has(c.gene)) { roundMain.push(c); assignedGenes.add(c.gene); }
      }
    }

    const roundGenes = [...roundRetry, ...roundMain].map(c => c.gene);
    const floor = computeExpectedRfsFloor(roundGenes);

    // Fallback genes = next batch_size from main pool not yet assigned
    const fallbacks = mainPool
      .filter(c => !assignedGenes.has(c.gene))
      .slice(0, batch_size)
      .map(c => c.gene);

    const rSkillGenes  = [...roundRetry, ...roundMain].filter(c => c.skill_rate !== null && c.skill_rate >= 0.6).map(c => c.gene);
    const rHypGenes    = [...roundRetry, ...roundMain].filter(c => c.signals.includes("speculative_hypothesis")).map(c => c.gene);
    const rRetryGenes  = roundRetry.map(c => c.gene);
    const rPlainGenes  = roundMain.filter(c => c.skill_rate === null && !c.signals.includes("speculative_hypothesis")).map(c => c.gene);

    roundPlans.push({
      round: r,
      genes: roundGenes,
      rationale: buildRationale(r, rSkillGenes, rHypGenes, rRetryGenes, rPlainGenes, floor),
      expected_rfs_floor: floor,
      fallback_genes: fallbacks,
    });

    mainIdx += roundMain.length;
    if (roundMain.length < mainBudget && retryIdx >= retryPool.length) break; // exhausted
  }

  const totalPlanned = roundPlans.reduce((n, r) => n + r.genes.length, 0);
  const summary = [
    `Planned ${totalPlanned} genes over ${roundPlans.length} rounds (batch_size=${batch_size})`,
    dataset ? `Dataset context: ${dataset}` : "No dataset context",
    retryGenes.size > 0 ? `${retryGenes.size} needs_investigation genes eligible for retry` : "No retry candidates",
    hypothesisGenes.size > 0 ? `${hypothesisGenes.size} genes with speculative hypotheses` : "No speculative hypotheses",
  ].join(". ");

  return {
    dataset,
    total_budget,
    batch_size,
    rounds: roundPlans,
    summary,
  };
}
