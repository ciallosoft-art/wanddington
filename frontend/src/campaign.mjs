// campaign.mjs — interactive oracle-backed perturbation experiment.
//
// The dataset's ground truth is held HIDDEN: each round the C-arm proposes a batch, the scientist
// commits, and only THEN does the oracle reveal the phenotype (which genes hit). Revealed hits/misses
// feed the next round (true sequential adaptation). This is the full experiment loop without wet-lab
// data — "挑选完 hit gene 才能看到扰动表型".

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { suggestGenes, revealBatch, ingestResults, REPO_ROOT } from "./brain.mjs";

const DEFAULT_ROUNDS = 5;

/**
 * Begin a campaign: cold-start recommendation for round 1, awaiting commit.
 * mode: "oracle" (benchmark truth plays the wet lab) | "upload" (scientist provides each round's readout).
 */
export async function startCampaign(state, { dataset, rounds, batchSize, mode }) {
  const r = await suggestGenes({ dataset, n: batchSize || undefined });
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
    startedAt: new Date().toISOString(),
  };
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
  const r = await ingestResults({ genes: c.pending, path, content, name, scoreCol, topRatio });
  const res = await advance(state, r.hits, r.totalHits);
  res.ingestMethod = r.method;
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
  c.totalHits = total; // stash so a later `stop` can still report cumulative/total
  const ratio = total ? cumulative / total : 0;
  c.history.push({ round: c.round, tested: c.pending.length, hits: roundHits, cumulative, ratio });

  const revealedRound = c.round;
  if (c.round < c.maxRounds) {
    c.round += 1;
    c.stage = "proposed";
    const next = await suggestGenes({
      dataset: c.dataset,
      n: c.batchSize || undefined,
      testedHits: c.testedHits,
      testedMisses: c.testedMisses,
    });
    c.pending = next.genes;
    return {
      kind: "campaign", stage: "revealed", done: false, mode: c.mode,
      dataset: c.dataset, round: revealedRound, roundHits, cumulative, total, ratio,
      nextRound: c.round, next: next.genes,
    };
  }
  return finish(state, { revealedRound, roundHits, cumulative, total, ratio });
}

/** End the campaign (natural end or user "stop"): summary + saved trace. */
export function finish(state, last = null) {
  const c = state.campaign;
  const lastRound = c.history.length ? c.history[c.history.length - 1] : null;
  const cumulative = lastRound ? lastRound.cumulative : c.testedHits.length;
  const totalHits = last?.total ?? c.totalHits ?? null;
  const ratio = totalHits ? cumulative / totalHits : lastRound?.ratio ?? 0;

  const trace = {
    dataset: c.dataset,
    rounds_run: c.history.length,
    batch_size: c.batchSize,
    started_at: c.startedAt,
    finished_at: new Date().toISOString(),
    history: c.history,
    cumulative_hits: cumulative,
    total_hits: totalHits,
    hit_ratio: Number(ratio.toFixed(4)),
  };
  let tracePath = null;
  try {
    const dir = join(REPO_ROOT, "workspace", "results", "campaigns");
    mkdirSync(dir, { recursive: true });
    tracePath = join(dir, `${c.dataset}-${c.startedAt.replace(/[:.]/g, "-")}.json`);
    writeFileSync(tracePath, JSON.stringify(trace, null, 2));
  } catch {
    /* trace is best-effort */
  }

  const result = {
    kind: "campaign", stage: "done", done: true,
    dataset: c.dataset,
    roundHits: last?.roundHits ?? [],
    cumulative, total: totalHits, ratio,
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
