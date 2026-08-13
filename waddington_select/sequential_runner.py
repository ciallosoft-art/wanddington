"""
SequentialRunner — executes N rounds of gene selection against an oracle.

Each round:
  1. arm.select(round_idx, revealed_so_far)  →  gene_list
  2. oracle.reveal(gene_list)                →  {gene: is_hit}
  3. arm.update(round_idx, new_reveals)
  4. accumulate metrics

Output is a RunResult with per-round hit counts, cumulative hit_ratio, and AUC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .arms.base import BaseArm
from .oracle import DatasetOracle


@dataclass
class RunResult:
    dataset: str
    arm_name: str
    batch_size: int
    n_rounds: int
    total_hits: int
    n_genes: int

    hits_per_round: list[int] = field(default_factory=list)
    cumulative_hits: list[int] = field(default_factory=list)
    hit_ratio_per_round: list[float] = field(default_factory=list)
    auc_normalized: float = 0.0
    revealed_hits: list[str] = field(default_factory=list)  # genes selected that were hits (for novel-hit analysis)
    # Gene-level audit trail. These make the direct basis for each hit recoverable without replaying
    # hundreds of remote calls: final tested batch, actual model names, and static padding per round.
    selected_genes_per_round: list[list[str]] = field(default_factory=list)
    llm_named_proposals_per_round: list[list[str]] = field(default_factory=list)
    llm_static_padding_proposals_per_round: list[list[str]] = field(default_factory=list)
    selected_llm_named_genes_per_round: list[list[str]] = field(default_factory=list)
    selected_without_llm_vote_per_round: list[list[str]] = field(default_factory=list)
    # Number of batch slots silently padded by the static ranker after the LLM failed to name a valid
    # in-pool gene. Empty for non-LLM arms. Recording this is essential when comparing providers.
    llm_fallback_per_round: list[int] = field(default_factory=list)
    # Provider/parser diagnostics without storing raw model text. Response hashes make repeated
    # outputs detectable while counts separate duplicates, already-tested genes, and out-of-pool
    # names from true JSON parse fallback.
    llm_diagnostics_per_round: list[dict] = field(default_factory=list)

    @property
    def final_hit_ratio(self) -> float:
        return self.hit_ratio_per_round[-1] if self.hit_ratio_per_round else 0.0


class SequentialRunner:
    """
    Drives one arm against one oracle for n_rounds rounds.

    Usage:
        runner = SequentialRunner(arm, oracle, n_rounds=5)
        result = runner.run()
    """

    def __init__(self, arm: BaseArm, oracle: DatasetOracle, n_rounds: int = 5) -> None:
        self.arm = arm
        self.oracle = oracle
        self.n_rounds = n_rounds

    def run(self, seed: int | None = None) -> RunResult:
        self.arm.reset()

        result = RunResult(
            dataset=self.oracle.dataset_name,
            arm_name=self.arm.name,
            batch_size=self.arm.batch_size,
            n_rounds=self.n_rounds,
            total_hits=self.oracle.total_hits,
            n_genes=self.oracle.n_genes,
        )

        revealed_all: dict[str, bool] = {}
        cumulative = 0

        for r in range(self.n_rounds):
            selected = self.arm.select(r, revealed_all)
            # Standalone LLMReasoningArm stores `_n_fallback` on itself, while hybrid C arms store it
            # on their nested `_llm` arm. Prefer the outer arm when it has the metric so we do not
            # accidentally inspect its low-level LLMClient instead.
            llm_component = (
                self.arm if hasattr(self.arm, "_n_fallback")
                else getattr(self.arm, "_llm", self.arm)
            )
            n_fallback = getattr(llm_component, "_n_fallback", None)
            has_llm = n_fallback is not None or hasattr(llm_component, "_llm_named")
            named = set(
                getattr(
                    self.arm,
                    "_last_llm_named_proposals",
                    getattr(llm_component, "_llm_named", set()),
                )
            )
            result.selected_genes_per_round.append(list(selected))
            static_padding = list(
                getattr(
                    self.arm,
                    "_last_llm_static_padding_proposals",
                    getattr(llm_component, "_static_padding", []),
                )
            ) if has_llm else []
            result.llm_named_proposals_per_round.append(sorted(named))
            result.llm_static_padding_proposals_per_round.append(static_padding)
            result.selected_llm_named_genes_per_round.append([gene for gene in selected if gene in named])
            result.selected_without_llm_vote_per_round.append(
                [gene for gene in selected if gene not in named] if has_llm else []
            )
            if n_fallback is not None:
                result.llm_fallback_per_round.append(int(n_fallback))
            diagnostics = getattr(llm_component, "_last_llm_diagnostics", None)
            if diagnostics is not None:
                result.llm_diagnostics_per_round.append(dict(diagnostics))
            new_reveals = self.oracle.reveal(selected)
            self.arm.update(r, new_reveals)
            revealed_all.update(new_reveals)

            round_hits = sum(new_reveals.values())
            cumulative += round_hits
            ratio = cumulative / self.oracle.total_hits if self.oracle.total_hits > 0 else 0.0

            result.hits_per_round.append(round_hits)
            result.cumulative_hits.append(cumulative)
            result.hit_ratio_per_round.append(round(ratio, 4))

        result.auc_normalized = self._normalized_auc(result)
        result.revealed_hits = [g for g, is_hit in revealed_all.items() if is_hit]
        return result

    @staticmethod
    def _normalized_auc(r: RunResult) -> float:
        """AUC of the hit_ratio curve, normalised by the optimal (oracle) AUC."""
        if r.total_hits == 0:
            return 0.0
        # Actual AUC (trapezoidal under cumulative hits curve, normalised by rounds)
        actual = sum(r.cumulative_hits) / (r.n_rounds * r.total_hits)
        # Optimal AUC: find hits as fast as possible each round
        opt_cumul = []
        found = 0
        for rnd in range(r.n_rounds):
            found = min(found + r.batch_size, r.total_hits)
            opt_cumul.append(found)
        optimal = sum(opt_cumul) / (r.n_rounds * r.total_hits)
        return round(actual / optimal, 4) if optimal > 0 else 0.0


def run_arm_on_dataset(
    arm: BaseArm,
    dataset_name: str,
    n_rounds: int = 5,
    seed: int = 42,
) -> RunResult:
    """Convenience wrapper: create oracle, run arm, return result."""
    oracle = DatasetOracle(dataset_name)
    runner = SequentialRunner(arm, oracle, n_rounds=n_rounds)
    return runner.run(seed=seed)
