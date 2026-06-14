import { planExperiments, type PlanResult } from "./experiment-planner.js";

// ---------------------------------------------------------------------------
// Argument parser
// ---------------------------------------------------------------------------

export interface SuggestGenesRequest {
  dataset?: string;
  budget: number;
  rounds?: number;
  batch_size?: number;
}

export function parseSuggestGenesArgs(raw: string): SuggestGenesRequest | { error: string } {
  // Strip leading /suggest-genes (with or without slash)
  const stripped = raw.replace(/^\/?\s*suggest-genes\s*/i, "").trim();
  const tokens = stripped.split(/\s+/).filter(Boolean);

  let dataset: string | undefined;
  let budget: number | undefined;
  let rounds: number | undefined;
  let batch_size: number | undefined;

  let i = 0;
  while (i < tokens.length) {
    const tok = tokens[i]!;
    if (tok === "--budget" || tok === "-b") {
      budget = parseInt(tokens[++i] ?? "", 10);
    } else if (tok === "--rounds" || tok === "-r") {
      rounds = parseInt(tokens[++i] ?? "", 10);
    } else if (tok === "--batch") {
      batch_size = parseInt(tokens[++i] ?? "", 10);
    } else if (tok === "--dataset" || tok === "-d") {
      dataset = tokens[++i]?.toUpperCase();
    } else if (!tok.startsWith("-")) {
      // First bare positional = dataset, second bare positional = budget
      if (!dataset) {
        dataset = tok.toUpperCase();
      } else if (!budget) {
        budget = parseInt(tok, 10);
      } else if (!rounds) {
        rounds = parseInt(tok, 10);
      }
    }
    i++;
  }

  if (!budget || budget < 1) {
    return { error: "budget is required and must be >= 1\nUsage: /suggest-genes [DATASET] --budget N [--rounds N] [--batch N]" };
  }

  return { dataset, budget, rounds, batch_size };
}

// ---------------------------------------------------------------------------
// Markdown formatter
// ---------------------------------------------------------------------------

export function formatPlanAsMarkdown(result: PlanResult): string {
  const lines: string[] = [];

  const heading = result.dataset
    ? `## Experiment Plan: ${result.dataset}`
    : "## Experiment Plan";
  lines.push(heading);
  lines.push(`_Budget: **${result.total_budget}** genes · Batch: **${result.batch_size}** · **${result.rounds.length}** rounds_`);
  lines.push("");
  lines.push(result.summary);
  lines.push("");

  for (const r of result.rounds) {
    lines.push(`### Round ${r.round}  (${r.genes.length} genes)`);
    lines.push(`> ${r.rationale}`);
    if (r.expected_rfs_floor > 0) {
      lines.push(`> Expected RFS floor: ≥ ${r.expected_rfs_floor}`);
    }
    lines.push("");
    // Print genes in a code block so they're copy-pasteable
    lines.push("```");
    lines.push(r.genes.join("  "));
    lines.push("```");
    if (r.fallback_genes.length > 0) {
      const shown = r.fallback_genes.slice(0, 8);
      const extra = r.fallback_genes.length - shown.length;
      lines.push(`_Fallbacks: ${shown.join(", ")}${extra > 0 ? ` (+${extra} more)` : ""}_`);
    }
    lines.push("");
  }

  // Total gene list for easy copy
  const allGenes = result.rounds.flatMap(r => r.genes);
  lines.push("---");
  lines.push(`**All ${allGenes.length} planned genes (copy-paste):**`);
  lines.push("```");
  lines.push(allGenes.join("  "));
  lines.push("```");

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Main handler (called from WebSocket handler and CLI)
// ---------------------------------------------------------------------------

export async function handleSuggestGenes(raw: string): Promise<string> {
  const parsed = parseSuggestGenesArgs(raw);

  if ("error" in parsed) {
    return `**Error:** ${parsed.error}`;
  }

  try {
    const result = await planExperiments({
      dataset:     parsed.dataset,
      total_budget: parsed.budget,
      rounds:      parsed.rounds,
      batch_size:  parsed.batch_size,
    });
    return formatPlanAsMarkdown(result);
  } catch (err) {
    return `**Error generating experiment plan:** ${(err as Error).message}`;
  }
}
