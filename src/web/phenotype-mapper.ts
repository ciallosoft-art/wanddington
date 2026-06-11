import { execFile } from "node:child_process";
import { resolve } from "node:path";
import { promisify } from "node:util";
import { getWaddingtonHome } from "../config/paths.js";
import { queryNeighbourhood } from "./knowledge-graph.js";
import { dbListHypotheses } from "./db.js";
import { findMatchingSkill } from "./skills.js";

const execFileAsync = promisify(execFile);

const MAPPER_SCRIPT = resolve(
  getWaddingtonHome(), "..", "workspace", "evaluation", "phenotype_mapper.py",
);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Pathway {
  id: string;
  name: string;
}

export interface PpiNeighbor {
  gene: string;
  score: number;
}

interface PythonResult {
  gene_name: string;
  gene_symbol: string;
  gene_summary: string;
  kegg_pathways: Pathway[];
  reactome_pathways: Pathway[];
  ppi_neighbors: PpiNeighbor[];
  error_pathways?: string;
  error_ppi?: string;
}

export interface KgEvidence {
  relation: string;   // e.g. "perturbed_in", "claims"
  target: string;     // dataset or metric label
  props: Record<string, unknown>;
}

export interface HypothesisSummary {
  hypothesis: string;
  confidence: "speculative" | "supported" | "refuted";
  evidence: string;
}

export interface SkillSummary {
  slug: string;
  success_rate: number;
  best_pearson_de?: number;
  run_count: number;
}

export interface PhenotypeProfile {
  gene: string;
  gene_name: string;
  gene_summary: string;

  // External API (MyGene.info + STRING)
  kegg_pathways: Pathway[];
  reactome_pathways: Pathway[];
  ppi_neighbors: PpiNeighbor[];

  // Internal (KG + hypotheses + SKILL)
  kg_evidence: KgEvidence[];
  hypotheses: HypothesisSummary[];
  skill?: SkillSummary;

  // Derived summary for LLM context injection
  context_hint: string;

  // Metadata
  errors: string[];
  cached_at: string;
}

// ---------------------------------------------------------------------------
// External API call (Python subprocess)
// ---------------------------------------------------------------------------

async function fetchExternalData(gene: string): Promise<PythonResult> {
  const { stdout } = await execFileAsync(
    "python3",
    [MAPPER_SCRIPT, gene.toUpperCase()],
    { timeout: 20_000 },
  );
  return JSON.parse(stdout) as PythonResult;
}

// ---------------------------------------------------------------------------
// KG evidence for this gene
// ---------------------------------------------------------------------------

function extractKgEvidence(gene: string): KgEvidence[] {
  const nodeIdOrLabel = gene.toUpperCase();
  const { edges, nodes } = queryNeighbourhood(nodeIdOrLabel, 2);

  const nodeMap = new Map(nodes.map(n => [n.id, n.label]));
  const evidence: KgEvidence[] = [];

  for (const edge of edges) {
    const srcLabel = nodeMap.get(edge.src) ?? edge.src;
    const dstLabel = nodeMap.get(edge.dst) ?? edge.dst;

    // Include edges where gene is either endpoint
    const geneUpper = gene.toUpperCase();
    if (srcLabel.toUpperCase() !== geneUpper && dstLabel.toUpperCase() !== geneUpper) continue;

    const target = srcLabel.toUpperCase() === geneUpper ? dstLabel : srcLabel;
    evidence.push({ relation: edge.rel, target, props: edge.props });
  }

  return evidence;
}

// ---------------------------------------------------------------------------
// Build context hint string (for LLM injection)
// ---------------------------------------------------------------------------

function buildContextHint(profile: Omit<PhenotypeProfile, "context_hint">): string {
  const lines: string[] = [`Gene: ${profile.gene} — ${profile.gene_name || "unknown function"}`];

  if (profile.gene_summary) {
    lines.push(`Summary: ${profile.gene_summary.slice(0, 200)}...`);
  }

  if (profile.kegg_pathways.length) {
    lines.push(`KEGG pathways: ${profile.kegg_pathways.map(p => p.name).slice(0, 3).join("; ")}`);
  }
  if (profile.reactome_pathways.length) {
    lines.push(`Reactome: ${profile.reactome_pathways.map(p => p.name).slice(0, 3).join("; ")}`);
  }
  if (profile.ppi_neighbors.length) {
    const top5 = profile.ppi_neighbors.slice(0, 5).map(n => n.gene).join(", ");
    lines.push(`Top PPI partners: ${top5}`);
  }

  const kgPerturbed = profile.kg_evidence.filter(e => e.relation === "perturbed_in");
  if (kgPerturbed.length) {
    lines.push(`Previously perturbed in: ${kgPerturbed.map(e => e.target).join(", ")}`);
  }

  const supported = profile.hypotheses.filter(h => h.confidence === "supported");
  const speculative = profile.hypotheses.filter(h => h.confidence === "speculative");
  if (supported.length) {
    lines.push(`Supported hypothesis: ${supported[0]!.hypothesis.slice(0, 120)}`);
  } else if (speculative.length) {
    lines.push(`Hypothesis (speculative): ${speculative[0]!.hypothesis.slice(0, 120)}`);
  }

  if (profile.skill) {
    lines.push(
      `SKILL: ${(profile.skill.success_rate * 100).toFixed(0)}% success over ${profile.skill.run_count} runs` +
      (profile.skill.best_pearson_de !== undefined
        ? `, best pearson_de=${profile.skill.best_pearson_de.toFixed(3)}`
        : ""),
    );
  }

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export async function getPhenotypeProfile(gene: string): Promise<PhenotypeProfile> {
  const errors: string[] = [];
  const geneUpper = gene.toUpperCase();

  // 1. External APIs (Python subprocess)
  let external: PythonResult = {
    gene_name: "", gene_symbol: geneUpper, gene_summary: "",
    kegg_pathways: [], reactome_pathways: [], ppi_neighbors: [],
  };
  try {
    external = await fetchExternalData(geneUpper);
    if (external.error_pathways) errors.push(`MyGene.info: ${external.error_pathways}`);
    if (external.error_ppi)      errors.push(`STRING PPI: ${external.error_ppi}`);
  } catch (err) {
    errors.push(`External API failed: ${(err as Error).message}`);
  }

  // 2. KG evidence
  let kgEvidence: KgEvidence[] = [];
  try {
    kgEvidence = extractKgEvidence(geneUpper);
  } catch (err) {
    errors.push(`KG query failed: ${(err as Error).message}`);
  }

  // 3. Hypotheses
  let hypotheses: HypothesisSummary[] = [];
  try {
    const rawHyps = dbListHypotheses({ gene: geneUpper, limit: 10 });
    hypotheses = rawHyps.map(h => ({
      hypothesis: h.hypothesis,
      confidence: h.confidence,
      evidence:   h.evidence,
    }));
  } catch (err) {
    errors.push(`Hypothesis query failed: ${(err as Error).message}`);
  }

  // 4. SKILL summary
  let skill: SkillSummary | undefined;
  try {
    const s = findMatchingSkill(geneUpper);
    if (s) {
      skill = {
        slug:           s.slug,
        success_rate:   s.success_rate,
        best_pearson_de: s.best_pearson_de,
        run_count:      s.run_count,
      };
    }
  } catch { /* SKILL lookup is optional */ }

  const partial = {
    gene: geneUpper,
    gene_name:         external.gene_name,
    gene_summary:      external.gene_summary,
    kegg_pathways:     external.kegg_pathways,
    reactome_pathways: external.reactome_pathways,
    ppi_neighbors:     external.ppi_neighbors,
    kg_evidence:       kgEvidence,
    hypotheses,
    skill,
    errors,
    cached_at: new Date().toISOString(),
  };

  return { ...partial, context_hint: buildContextHint(partial) };
}
