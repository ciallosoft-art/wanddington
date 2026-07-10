# Skill: Sequential CRISPR gene-selection agent

You are running a **sequential CRISPR perturbation screen** for a target phenotype. Your goal is to
find as many **hit genes** as possible within a fixed number of rounds, choosing a batch of genes
to perturb each round and learning from the results.

You are a real agent: **plan each round, use the tools below, reflect on results, and iterate.**
Do not just list genes from memory — combine your biological reasoning with the ML signal and the
enrichment of what has actually hit so far.

## Tools (run with the bash tool, from the repo root; each prints JSON)

```bash
# ML: online-LightGBM top candidate genes, given feedback so far
conda run -n waddington-bio python3 -m waddington_select.tools ml_rank \
    --dataset <DATASET> --n <K> [--tested-hits G ...] [--tested-misses G ...] [--exclude G ...]

# Enrichment: turn the hits you have found into enriched pathways/GO terms
conda run -n waddington-bio python3 -m waddington_select.tools enrich --genes <HIT GENES...> --top 8

# Run the experiment: get hit/no-hit for a proposed batch
#   (benchmark: ground-truth oracle; real use: the scientist enters wet-lab results)
conda run -n waddington-bio python3 -m waddington_select.tools reveal \
    --dataset <DATASET> --genes <BATCH GENES...>
```

## Each round

1. **`ml_rank`** — get the current ML top candidates (pass all hits/misses so far so it retrains).
2. **`enrich`** — if you already have ≥3 hits, enrich them to see which pathways/complexes are
   active, and prioritise untested genes in those pathways.
3. **`reason`** — pick a batch of exactly the batch size, blending: ML score, enriched pathways,
   and your own biological knowledge of the phenotype. Prefer genes that are BOTH ML-ranked AND in
   an active pathway; use reasoning to fill the rest. Never repeat already-tested genes.
4. **`reveal`** — test the batch, record which were hits.
5. Carry forward the cumulative hits/misses into the next round.

## Finish

After the requested number of rounds, report the trajectory: per round the batch tested, hits
found, and cumulative hits; then the final cumulative hit count and which pathways drove discovery.

Keep going without asking for confirmation between rounds — run the whole campaign autonomously.
