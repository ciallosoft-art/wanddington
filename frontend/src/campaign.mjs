// campaign.mjs — interactive oracle-backed perturbation experiment.
//
// The dataset's ground truth is held HIDDEN: each round the C-arm proposes a batch, the scientist
// commits, and only THEN does the oracle reveal the phenotype (which genes hit). Revealed hits/misses
// feed the next round (true sequential adaptation). This is the full experiment loop without wet-lab
// data — "挑选完 hit gene 才能看到扰动表型".

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";

import { suggestGenes, revealBatch, ingestResults, REPO_ROOT } from "./brain.mjs";

const DEFAULT_ROUNDS = 5;

/**
 * Begin a campaign: cold-start recommendation for round 1, awaiting commit.
 * mode: "oracle" (benchmark truth plays the wet lab) | "upload" (scientist provides each round's readout).
 */
/** suggest() traces its own single selection; renumber it to the campaign's round. */
function pushDecision(c, info, round) {
  const d = info?.trace;
  if (!d) return;
  d.round = round;
  c.decisions.push(d);
}

/** After a reveal, mark which of that round's picks were hits — decisions + outcomes in one artifact. */
function attachOutcome(c, round, roundHits) {
  const d = c.decisions.find((x) => x.round === round);
  if (!d) return;
  const hs = new Set(roundHits);
  for (const g of d.genes) g.hit = hs.has(g.gene);
  d.n_hits = roundHits.length;
}

export async function startCampaign(state, { dataset, rounds, batchSize, mode }) {
  const r = await suggestGenes({ dataset, n: batchSize || undefined, trace: true });
  state.campaign = {
    active: true,
    mode: mode === "upload" ? "upload" : "oracle",
    dataset,
    batchSize: batchSize || null,
    round: 1,
    maxRounds: rounds && rounds > 0 ? rounds : DEFAULT_ROUNDS,
    stage: "proposed",
    pending: r.genes,
    testedHits: [],
    testedMisses: [],
    history: [],
    decisions: [],   // why each round's genes were chosen (attribution + SHAP)
    startedAt: new Date().toISOString(),
  };
  pushDecision(state.campaign, r.info, 1);
  return {
    kind: "campaign", stage: "proposed", mode: state.campaign.mode,
    dataset, round: 1, maxRounds: state.campaign.maxRounds, genes: r.genes,
  };
}

/**
 * Commit the pending batch.
 * - oracle mode: reveal via the hidden ground truth now, then advance.
 * - upload mode: do NOT reveal — wait for the scientist to provide this round's screen readout.
 */
export async function commitRound(state) {
  const c = state.campaign;
  if (c.stage === "recommendation_pending") return retryRecommendation(state);
  if (c.mode === "upload") {
    c.stage = "awaiting_results";
    return {
      kind: "campaign", stage: "awaiting_results", mode: "upload",
      dataset: c.dataset, round: c.round, genes: c.pending,
    };
  }
  const reveal = await revealBatch({ dataset: c.dataset, genes: c.pending });
  return advance(state, reveal.hits, reveal.totalHits);
}

/** Upload mode: the scientist provides this round's readout file → derive hits → advance. */
export async function submitResults(state, { path, content, name, scoreCol, topRatio } = {}) {
  const c = state.campaign;
  if (c.stage === "recommendation_pending") return retryRecommendation(state);
  const r = await ingestResults({ genes: c.pending, path, content, name, scoreCol, topRatio, dataset: c.dataset });
  if (r.unknown?.length) {
    throw new Error(`Results are missing tested genes: ${r.unknown.join(", ")}. Provide a complete readout before committing this round.`);
  }
  const res = await advance(state, r.hits, r.totalHits);
  res.ingestMethod = r.method;
  res.hitRule = r.rule;
  res.scope = r.scope;
  res.overridesIgnored = r.overridesIgnored;
  res.unknown = r.unknown;
  return res;
}

/** Record a round's outcome (hits known), then propose the next round or finish. Shared by both modes. */
async function advance(state, revealedHits, total) {
  const c = state.campaign;
  const hitSet = new Set(revealedHits);
  const roundHits = c.pending.filter((g) => hitSet.has(g));
  const roundMisses = c.pending.filter((g) => !hitSet.has(g));

  c.testedHits.push(...roundHits);
  c.testedMisses.push(...roundMisses);
  const cumulative = c.testedHits.length;
  // The screen-wide hit count is known only when the whole screen's readout was uploaded (or the oracle
  // plays the wet lab). For one batch per round it is unknown: report hits per tested gene instead.
  c.totalHits = total ?? null; // stash so a later `stop` can still report cumulative/total
  const ratio = total ? cumulative / total : null;
  const testedSoFar = c.testedHits.length + c.testedMisses.length;
  const hitRate = testedSoFar ? cumulative / testedSoFar : 0;
  c.history.push({ round: c.round, tested: c.pending.length, hits: roundHits, cumulative, ratio, testedSoFar, hitRate });

  const revealedRound = c.round;
  attachOutcome(c, revealedRound, roundHits);

  if (c.round < c.maxRounds) {
    // The outcome is durable before the next API call. A retry must only rank again,
    // never reveal/upload the same batch twice or increment the round prematurely.
    c.stage = "recommendation_pending";
    c.pendingOutcome = { revealedRound, roundHits, cumulative, total: c.totalHits, ratio, testedSoFar, hitRate };
    return retryRecommendation(state);
  }
  return finish(state, { revealedRound, roundHits, cumulative, total: c.totalHits, ratio, testedSoFar, hitRate });
}

/** Retry selection after a saved outcome, without replaying that round's results. */
export async function retryRecommendation(state) {
  const c = state.campaign;
  const outcome = c.pendingOutcome;
  let next;
  try {
    next = await suggestGenes({
      dataset: c.dataset,
      n: c.batchSize || undefined,
      testedHits: c.testedHits,
      testedMisses: c.testedMisses,
      trace: true,
    });
  } catch (error) {
    // Return state even on failure so the stateless Web transport can persist the
    // confirmed outcomes. Both existing renderers already display an `await` turn.
    return {
      kind: "campaign", stage: "await", mode: c.mode, dataset: c.dataset,
      round: c.round, roundHits: outcome.roundHits,
      cumulative: outcome.cumulative, total: outcome.total, ratio: outcome.ratio,
      testedSoFar: outcome.testedSoFar, hitRate: outcome.hitRate,
      text: `Round ${c.round}'s results are saved (${outcome.roundHits.length} hits; cumulative ${outcome.cumulative}). The next recommendation failed: ${error.message || error}. Type \`commit\` to retry, or \`stop\` to finish with the saved results.`,
    };
  }
  c.round = outcome.revealedRound + 1;
  c.stage = "proposed";
  c.pending = next.genes;
  pushDecision(c, next.info, c.round);
  delete c.pendingOutcome;
  return {
    kind: "campaign", stage: "revealed", done: false, mode: c.mode,
    dataset: c.dataset, round: outcome.revealedRound,
    roundHits: outcome.roundHits, cumulative: outcome.cumulative,
    total: outcome.total, ratio: outcome.ratio,
    testedSoFar: outcome.testedSoFar, hitRate: outcome.hitRate,
    nextRound: c.round, next: next.genes,
  };
}

/** End the campaign (natural end or user "stop"): summary + saved trace. */
export function finish(state, last = null) {
  const c = state.campaign;
  const lastRound = c.history.length ? c.history[c.history.length - 1] : null;
  const cumulative = lastRound ? lastRound.cumulative : c.testedHits.length;
  const totalHits = last?.total ?? c.totalHits ?? null;
  const ratio = totalHits ? cumulative / totalHits : null;
  const testedSoFar = c.testedHits.length + c.testedMisses.length;
  const hitRate = testedSoFar ? cumulative / testedSoFar : 0;

  const trace = {
    dataset: c.dataset,
    rounds_run: c.history.length,
    batch_size: c.batchSize,
    started_at: c.startedAt,
    finished_at: new Date().toISOString(),
    history: c.history,
    cumulative_hits: cumulative,
    total_hits: totalHits,
    hit_ratio: ratio == null ? null : Number(ratio.toFixed(4)),   // null: screen-wide hit count unknown
    tested_genes: testedSoFar,
    hits_per_tested: Number(hitRate.toFixed(4)),
    // decisions + outcomes → feed straight to:
    //   python -m waddington_select.analysis report --campaign <this file>
    trace: { rounds: (c.decisions || []).filter((d) => d.genes?.some((g) => "hit" in g)) },
  };
  let tracePath = null;
  try {
    const dir = join(REPO_ROOT, "workspace", "results", "campaigns");
    mkdirSync(dir, { recursive: true });
    // Dataset and timestamps can arrive in client-supplied Web state. Keep them
    // inside the artifact; neither may determine its filesystem location.
    const destination = join(dir, `${randomUUID()}.json`);
    writeFileSync(destination, JSON.stringify(trace, null, 2));
    tracePath = destination;
  } catch {
    /* trace is best-effort */
  }

  const result = {
    kind: "campaign", stage: "done", done: true,
    dataset: c.dataset,
    roundHits: last?.roundHits ?? [],
    cumulative, total: totalHits, ratio, testedSoFar, hitRate,
    history: c.history, tracePath,
  };
  state.campaign = null; // end
  return result;
}

/** Classify a message inside an active campaign (deterministic; EN + 中文). */
export function campaignCommand(message) {
  const m = message.trim().toLowerCase();
  if (/^(commit|yes|y|ok|okay|go|next|test|confirm|proceed)\b/.test(m) ||
      /确认|继续|提交|下一轮|开始测|测试|下单/.test(message)) return "commit";
  if (/\b(stop|end|quit|exit|abort|done)\b/.test(m) || /结束|停止|停|退出|终止/.test(message)) return "stop";
  return "other";
}
