# Results: Frozen Paper Outputs and Active GPT Runs

The paper's frozen numbers and figures are generated from the historical JSONs by
`python -m waddington_select.analysis` — no value is transcribed by hand. Active GPT experiments live
under `gpt/`; they are separate experiments and do not replace or relabel paper results.

## Layout

```
results/
├── sequential/     # frozen supporting runs + historical Claude memory
├── router/         # frozen paper headline/router controls + protocol.json
├── gpt/
│   ├── gpt_5_6_luna_none/     # formal staged GPT calibration/memory/C-arm outputs
│   └── gpt_5_6_luna_minimal/  # superseded incomplete pilot; not a headline
├── figures/        # generated figures (waddington_select.analysis.figures)
├── bda_controlled/ # controlled BioDiscoveryAgent run on our benchmark
├── prior_probes.json          # what the cross-experiment prior learns (sibling / feature-family / hit-freq / anchor)
├── final_system_stats.json    # final 20-seed headline/paired stats + exploratory 5-seed novel-hit stats
├── conditional_attribution.json  # LLM-endorsement odds ratio (verifier analysis)
├── clustered_ci.json          # screen-clustered bootstrap CIs
├── novel_hit_analysis.json    # recurrent vs novel hit recall
└── attribution_9ds.json       # pooled per-source attribution (produced by analysis/run_attribution.py)
```

`campaigns/` (interactive-frontend traces) is gitignored.

The formal GPT directory is complete only when `main_table_5rep.json` and its manifest both exist
with `status: complete`. Its LLM conditions are five repeated API invocations (`provider_seed: null`),
because the Codex OAuth endpoint does not expose a sampling seed; independence between API calls is
not assumed. In the formal six-arm set, only Random consumes the local replicate-index seed.
