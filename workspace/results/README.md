# Results

Frozen benchmark outputs. The paper's numbers and figures are generated from these JSONs by
`python -m waddington_select.analysis` — no value is transcribed by hand.

## Layout

```
results/
├── sequential/     # main benchmark: run_sequential.py output (per arm × dataset × seed) + experience_memory.json
├── router/         # leakage-free router / final-system runs + router_analysis, protocol.json
├── figures/        # generated figures (waddington_select.analysis.figures)
├── bda_controlled/ # controlled BioDiscoveryAgent run on our benchmark
├── prior_probes.json          # what the cross-experiment prior learns (sibling / feature-family / hit-freq / anchor)
├── final_system_stats.json    # final 5-seed system: matched-LOO, paired deltas, novel-hit stats
├── conditional_attribution.json  # LLM-endorsement odds ratio (verifier analysis)
├── clustered_ci.json          # screen-clustered bootstrap CIs
├── novel_hit_analysis.json    # recurrent vs novel hit recall
└── attribution_9ds.json       # pooled per-source attribution (produced by analysis/run_attribution.py)
```

`campaigns/` (interactive-frontend traces) is gitignored.

## Legacy

`runs/` and `summary.csv` are outputs of the earlier V6 **static-ranker** evaluation
(`workspace/evaluation/benchmark.py`, viewed with `results_summary.py`), superseded by `sequential/`.
They are kept as a historical record.
