import { dbListExperiments } from "./db.js";
import type { RFSResult } from "./rfs.js";
import type { FailureRecord } from "./failure.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type FilterVerdict =
  | "technical_failure"     // experiment ran incorrectly — retry after fix
  | "true_negative"         // experiment ran correctly; gene has no effect on phenotype
  | "needs_investigation";  // ambiguous; insufficient signal or contradictory evidence

export interface FilterResult {
  verdict: FilterVerdict;
  confidence: "high" | "medium" | "low";
  reason: string;
  action: "retry_with_fix" | "remove_from_candidates" | "manual_review";
  suggested_fix?: string;   // forwarded from E3 FailureRecord when relevant
  supporting_evidence: string[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// E3 failure types that unambiguously indicate a technical problem
const TECHNICAL_FAILURE_TYPES = new Set([
  "environment_failure",
  "data_access_failure",
  "protocol_ambiguity",
]);

// RFS thresholds
const PROTOCOL_FIDELITY_MIN = 0.5;   // below this → code deviated from protocol
const OVERALL_RFS_GOOD      = 0.65;  // above this → experiment quality sufficient
const PROTOCOL_RFS_GOOD     = 0.7;   // above this → code followed protocol well
const BIOLOGY_SCORE_OK      = 0.6;   // above this → DE genes have pathway signal

// Metric thresholds (below these → "low performance" for primary metrics)
const LOW_METRIC_THRESHOLDS: Record<string, number> = {
  pearson_de:    0.30,
  pearson:       0.20,
  pearson_delta: 0.20,
  r2:            0.10,
};

// How many past high-quality runs we need to confirm a true negative
const MIN_QUALITY_RUNS_FOR_NEGATIVE = 2;

// ---------------------------------------------------------------------------
// Helper: is the primary metric low?
// ---------------------------------------------------------------------------

function isPrimaryMetricLow(metrics: Record<string, number>): boolean {
  for (const [name, threshold] of Object.entries(LOW_METRIC_THRESHOLDS)) {
    const val = metrics[name];
    if (val !== undefined) return val < threshold;
  }
  // No recognised metric present — cannot judge
  return false;
}

// ---------------------------------------------------------------------------
// Helper: count past high-quality runs with consistently low metrics
// ---------------------------------------------------------------------------

function countQualityRunsWithLowMetrics(gene: string, currentMetrics: Record<string, number>): number {
  const history = dbListExperiments({ gene: gene.toUpperCase(), limit: 20 });
  let count = 0;

  for (const exp of history) {
    if (!exp.rfs_json) continue;
    let rfs: RFSResult;
    try { rfs = JSON.parse(exp.rfs_json) as RFSResult; } catch { continue; }

    // Only count experiments that ran correctly
    if (rfs.overall < OVERALL_RFS_GOOD || rfs.protocol < PROTOCOL_RFS_GOOD) continue;

    // Check if metrics were also low in that run
    let expMetrics: Record<string, number> = {};
    try { expMetrics = JSON.parse(exp.metrics || "{}") as Record<string, number>; } catch { continue; }
    if (isPrimaryMetricLow(expMetrics)) count++;
  }

  // Also count the current run if it has good RFS
  void currentMetrics; // counted by caller after verdict decision
  return count;
}

// ---------------------------------------------------------------------------
// Main function
// ---------------------------------------------------------------------------

export function applyNegativeFilter(
  gene: string,
  metrics: Record<string, number>,
  rfs: RFSResult,
  failure?: FailureRecord | null,
): FilterResult {
  const evidence: string[] = [];

  // ── Path 1: clear technical failure via E3 type ──────────────────────────
  if (failure && TECHNICAL_FAILURE_TYPES.has(failure.type)) {
    evidence.push(`E3 failure type: ${failure.type}`);
    evidence.push(...failure.evidence.slice(0, 2));
    return {
      verdict: "technical_failure",
      confidence: "high",
      reason: `${failure.message} (E3 classification: ${failure.type})`,
      action: "retry_with_fix",
      suggested_fix: failure.suggested_fix,
      supporting_evidence: evidence,
    };
  }

  // ── Path 2: protocol fidelity too low → code deviated from spec ──────────
  if (rfs.protocol >= 0 && rfs.protocol < PROTOCOL_FIDELITY_MIN) {
    evidence.push(`RFS.protocol = ${rfs.protocol.toFixed(2)} < ${PROTOCOL_FIDELITY_MIN}`);
    if (rfs.details.protocol_issues.length) {
      evidence.push(...rfs.details.protocol_issues.slice(0, 2));
    }
    return {
      verdict: "technical_failure",
      confidence: "high",
      reason: `Protocol fidelity ${rfs.protocol.toFixed(2)} is too low — experiment did not follow paper protocol`,
      action: "retry_with_fix",
      suggested_fix: "Run /paper-audit to re-extract ProtocolSpec and fix normalization / n_hvg / split settings",
      supporting_evidence: evidence,
    };
  }

  // ── Path 3: overall RFS low but biology is fine → technical issue likely ──
  if (
    rfs.overall >= 0 && rfs.overall < 0.40 &&
    rfs.biology >= 0 && rfs.biology > BIOLOGY_SCORE_OK
  ) {
    evidence.push(`RFS.overall = ${rfs.overall.toFixed(2)} (low)`);
    evidence.push(`RFS.biology = ${rfs.biology.toFixed(2)} (OK — DE genes have pathway signal)`);
    evidence.push("Discrepancy suggests code error, not biological absence");
    return {
      verdict: "technical_failure",
      confidence: "medium",
      reason: "Low overall performance despite normal GO/PPI enrichment — likely a technical issue rather than true negative",
      action: "retry_with_fix",
      suggested_fix: failure?.suggested_fix ?? "Review metric computation and data preprocessing steps",
      supporting_evidence: evidence,
    };
  }

  // ── Path 4: good experiment quality → evaluate biological signal ──────────
  const experimentQualityOk =
    rfs.overall >= OVERALL_RFS_GOOD &&
    (rfs.protocol < 0 || rfs.protocol >= PROTOCOL_RFS_GOOD); // -1 means no spec = skip check

  if (experimentQualityOk && isPrimaryMetricLow(metrics)) {
    evidence.push(`RFS.overall = ${rfs.overall.toFixed(2)} (good quality)`);
    if (rfs.protocol >= 0) evidence.push(`RFS.protocol = ${rfs.protocol.toFixed(2)} (protocol followed)`);

    const priorLowQualityRuns = countQualityRunsWithLowMetrics(gene, metrics);
    evidence.push(`Prior high-quality runs with low metrics: ${priorLowQualityRuns}`);

    if (priorLowQualityRuns >= MIN_QUALITY_RUNS_FOR_NEGATIVE) {
      return {
        verdict: "true_negative",
        confidence: "high",
        reason: `${priorLowQualityRuns + 1} high-quality experiments (RFS ≥ ${OVERALL_RFS_GOOD}) all show low performance — gene likely has no effect on this phenotype`,
        action: "remove_from_candidates",
        supporting_evidence: evidence,
      };
    }

    if (priorLowQualityRuns === 1) {
      return {
        verdict: "true_negative",
        confidence: "medium",
        reason: `2 high-quality experiments show low performance — likely true negative, but one more run would confirm`,
        action: "remove_from_candidates",
        supporting_evidence: evidence,
      };
    }

    // Only 1 good run — not enough to confirm
    return {
      verdict: "needs_investigation",
      confidence: "low",
      reason: "Experiment quality is good and metrics are low, but only 1 high-quality run — need at least one more replicate to confirm true negative",
      action: "manual_review",
      supporting_evidence: evidence,
    };
  }

  // ── Path 5: metric_mismatch — neither biological nor clear technical ──────
  if (failure?.type === "metric_mismatch") {
    evidence.push(`E3 failure type: metric_mismatch`);
    evidence.push(...failure.evidence.slice(0, 2));
    return {
      verdict: "technical_failure",
      confidence: "medium",
      reason: "Metric definition differs from paper specification — result is not comparable to paper's reported values",
      action: "retry_with_fix",
      suggested_fix: failure.suggested_fix,
      supporting_evidence: evidence,
    };
  }

  // ── Path 6: biology suspicious with otherwise ok RFS ─────────────────────
  if (failure?.type === "biology_suspicious" && experimentQualityOk) {
    evidence.push("E3: biology_suspicious — DE genes do not match expected pathway priors");
    if (rfs.biology >= 0) evidence.push(`RFS.biology = ${rfs.biology.toFixed(2)}`);
    return {
      verdict: "needs_investigation",
      confidence: "medium",
      reason: "Experiment quality is acceptable but biological signal is suspicious — possible batch effect or data leakage",
      action: "manual_review",
      suggested_fix: failure.suggested_fix,
      supporting_evidence: evidence,
    };
  }

  // ── Default ───────────────────────────────────────────────────────────────
  evidence.push(`RFS.overall = ${rfs.overall >= 0 ? rfs.overall.toFixed(2) : "n/a"}`);
  evidence.push(`RFS.protocol = ${rfs.protocol >= 0 ? rfs.protocol.toFixed(2) : "n/a (no spec)"}`);
  return {
    verdict: "needs_investigation",
    confidence: "low",
    reason: "Insufficient signal to classify — more experiments or manual review needed",
    action: "manual_review",
    supporting_evidence: evidence,
  };
}

// ---------------------------------------------------------------------------
// Formatting helper (for Pi context injection)
// ---------------------------------------------------------------------------

export function formatFilterResult(r: FilterResult): string {
  const lines = [
    `NegativeFilter verdict: ${r.verdict.toUpperCase()} (confidence: ${r.confidence})`,
    `Reason: ${r.reason}`,
    `Action: ${r.action}`,
  ];
  if (r.suggested_fix) lines.push(`Fix: ${r.suggested_fix}`);
  return lines.join("\n");
}
