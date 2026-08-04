# Waddington

**A hybrid ML + LLM + verified-memory agent for sequential CRISPR gene selection.**

Named after Conrad Waddington (1905–1975) and his epigenetic landscape. Given a target
phenotype and a genome-scale pool of candidate genes, the agent picks — over a few rounds of
CRISPR-screen feedback — which genes to perturb next, aiming to find "hit" genes as fast as
possible. This is the **BDA-style active gene-selection** problem (BioDiscoveryAgent /
PerTurboAgent lineage), *not* perturbation-effect prediction.

---

## The agent (C-arm)

`waddington_select` implements a three-component selector that runs a 5-round sequential loop
against an oracle. Each round it selects a batch of genes, the oracle reveals hit/non-hit, and
the feedback conditions the next round.

| Component | What it does |
|-----------|--------------|
| **Online adaptive ML** (`OnlineAdaptiveArm`, LightGBM) | Scores every gene from PPI / KEGG / pLI / DepMap features; **retrains each round** on revealed feedback. |
| **LLM reasoning** (`LLMReasoningArm`, Claude Haiku, T=0) | Names genes from parametric biology, conditioned on task, feedback, and cross-experiment memory. Names are matched to the real gene pool; unmatched slots fall back to the LOO ranker. |
| **Cross-experiment memory** (`memory_builder`) | Strategy summaries distilled from the *other* datasets (leave-one-out), injected into the LLM prompt. Uses **DeLM-style verified admission** — a mechanical strategy-label check plus an LLM verifier that rejects hallucinated gene names before an entry is written. |

**Fusion.** The reported (leakage-free) system uses a single **global** weighted fusion — ML 0.80 /
LLM 0.20 — on every screen, with the weight fixed on non-target screens. Unmatched LLM slots fall back
to the LOO ranker.

> **Legacy target-aware router (analyzed, not reported).** An earlier variant
> (`waddington_c_arm._classify`) picks the fusion mode from the target screen's *realized* hit rate —
> `ml_heavy` (ML 0.80 / LLM 0.20), `baseline` (0.60 / 0.40), or `two_stage` (ML shortlists 384, LLM
> re-ranks). Because that reads a test-set statistic, it is a **leakage** variant kept only for the
> §honest-router analysis (avg 0.256, no detected difference from the reported 0.251). It is **not** the
> system we report. Enable it (and the metadata feature policy / no-anchor-features) via the flags in
> **Reproduce the benchmark** below.

---

## Results

Hit ratio at round 5, mean over 5 seeds × 9 BDA benchmark datasets (DeLM-verified memory):

| Method | avg hit@R5 |
|--------|-----------|
| Random | 0.066 |
| A: Coreset | 0.134 |
| LOO-LightGBM (static) | 0.217 |
| OnlineAdaptive | 0.224 |
| B: LLM reasoning | 0.225 |
| **C: Waddington (ours, leakage-free)** | **0.251** |

The reported system is the **leakage-free, privilege-free** configuration (0.251): a *global* fusion
weight (`w_llm = 0.2`, chosen on non-target screens — never the target's realized hit rate), a
metadata-only DepMap feature policy, and **no** anchor-relative features. An earlier **target-aware
routed** variant scores 0.256, but it reads the target screen's realized hit rate to pick its fusion
mode (a test-set statistic), so we do **not** report it as the system — the two are statistically
indistinguishable (paired −0.005, 95% CI [−0.018, 0.007]). See `§honest-router` in the paper.

> **Where the advantage comes from (be honest):** the dominant ingredient is the **cross-experiment
> supervised prior**, not agentic reasoning. A static LOO-LightGBM with *no* LLM already reaches 0.187
> on the six screens shared with BioDiscoveryAgent — nearly double BDA's LLM agent (0.105) and
> GeneDisco's active learning (0.113). Online adaptation and constrained LLM reasoning add a further,
> smaller gain (against its *matched* static prior, 0.231 → 0.251, +0.020, 95% CI [0.007, 0.035]).
> On genuinely *novel* (non-recurrent) hits the full system does **not** reliably beat the prior.

Full per-dataset tables, the ablation study (memory / LLM / online-ML / gene-name-shuffle), the
leakage-free router analysis, and the architecture progression are in
[`docs/results_tables.tex`](docs/results_tables.tex).

---

## Layout

```
waddington_select/        # the agent package (the "brain", Python)
├── run_sequential.py     #   BENCHMARK: experiment runner (python -m waddington_select)
├── agent_benchmark.py    #   BENCHMARK: tool-using agent vs pipeline (documented negative result)
├── sequential_runner.py  #   the 5-round oracle loop
├── oracle.py             #   BDA benchmark oracle + dataset registry
├── suggest.py            #   DEPLOY: forward recommendation (no oracle; --json for the frontend)
├── simulate.py           #   DEPLOY: narrated oracle-driven demo campaign
├── memory_builder.py     #   cross-experiment memory w/ DeLM verified admission
└── arms/                 #   selection arms (C-arm, baselines, ablations, archive/ = dev history)

frontend/                 # the conversational entry (CLI + Web, Node) — see frontend/README.md
├── bin/waddington.js     #   `node bin/waddington.js` → auth → CLI/Web → chat
└── src/                  #   tool-less pi-ai bridge + intent routing; drives suggest/simulate

workspace/
├── evaluation/           # feature pipeline + LOO LightGBM training data
├── models/               # trained per-dataset LightGBM models
├── data/bda_benchmark/   # task prompts + benchmark labels
└── results/sequential/   # benchmark result JSONs + experience_memory.json

docs/                     # research iteration history (v1..v26) + results_tables.tex
experiments/              # reproduction scripts (01_baselines / 02_three_arm / 03_ablations)
```

---

## Setup (fresh clone)

Two halves: a Python **brain** and a Node **frontend**. All runtime data (features, PPI/ARCHS4
caches, task prompts, memory, models) is committed, so a clone only needs the two toolchains.

```bash
# 1. Brain — creates the `waddington-bio` conda env and installs the package
conda env create -f environment.yml
conda activate waddington-bio

# 2. Frontend — Node 20–24
cd frontend && npm install && cd ..

# 3. Sanity checks
conda run -n waddington-bio python3 -m waddington_select.suggest --dataset IFNG --n 5   # brain
node frontend/bin/waddington.js --help                                                  # frontend
```

No API key needs to be pre-configured: on first launch the frontend prompts you to authorize a
provider (Claude / Codex / Gemini) via OAuth — exactly like pi/feynman — or set an API-key env var.

## Conversational entry (scientists)

The primary way to *use* the agent — natural-language chat that drives the C-arm pipeline:

```bash
cd frontend && node bin/waddington.js      # authorize → pick Terminal (CLI) or Web UI → chat
```

See [`frontend/README.md`](frontend/README.md). Or drive the brain directly:

```bash
# forward recommendation (real experiment; feed back results to adapt the next round)
conda run -n waddington-bio python3 -m waddington_select.suggest --dataset IFNG \
    --tested-hits STAT1 JAK2 --tested-misses ACTB GAPDH
# narrated demo campaign (oracle plays the wet lab)
conda run -n waddington-bio python3 -m waddington_select.simulate --dataset IFNG --rounds 5
```

## Reproduce the benchmark

The benchmark is kept separate from deployment and frozen on `claude-haiku` for reproducibility.

```bash
bash experiments/01_baselines.sh     # no API needed
bash experiments/02_three_arm.sh     # A / B / C
bash experiments/03_ablations.sh     # four ablations

# Or run the runner directly (from repo root)
conda run -n waddington-bio python3 -m waddington_select \
    --arms coreset llm_reasoning waddington_c --seeds 5 --rounds 5

# Rebuild the cross-experiment memory (with verified admission)
conda run -n waddington-bio python3 -m waddington_select.memory_builder
```

**Reported (leakage-free) vs legacy config.** By default `waddington_c` **is** the reported
leakage-free system (avg 0.251): global `w_llm = 0.2`, metadata feature policy, no anchor features —
none of which read the target screen's realized labels. No flags needed:

```bash
# Reported leakage-free C (0.251) — the plain default
conda run -n waddington-bio python3 -m waddington_select --arms waddington_c --seeds 5 --rounds 5

# Final-system paired statistics (matched-LOO, vs baselines, novel-hit)
conda run -n waddington-bio python3 -m waddington_select.analysis.final_system_stats
```

To reproduce the **legacy target-aware routed** configuration (avg 0.256, reads the realized hit rate;
used for the legacy SHAP / decomposition / ablation tables), set one flag:

```bash
WADDINGTON_LEGACY_ROUTER=1 conda run -n waddington-bio python3 -m waddington_select \
    --arms waddington_c --seeds 5 --rounds 5
```

(`experiments/03_ablations.sh` sets `WADDINGTON_LEGACY_ROUTER=1` for you; `02_three_arm.sh` produces
both configurations.)

Arm names: `random`, `coreset`, `static_ranker`, `online_adaptive`, `llm_reasoning`,
`waddington_c`, and the ablations `waddington_c_no_memory` / `_no_llm` / `_no_ml` /
`_shuffled_names`.

## Configuration

Environment variables (all optional; sensible defaults):

| Variable | Default | Purpose |
|----------|---------|---------|
| `WADDINGTON_AUTH_PATH` | `~/.waddington/agent/auth.json` (own store; `WADDINGTON_REUSE_FEYNMAN=1` reuses `~/.feynman`) | OAuth/API token store used by the frontend |
| `WADDINGTON_PY` | `conda run -n waddington-bio python3` | How the frontend invokes the Python brain |
| `WADDINGTON_CHAT_MODEL` | an authorized `anthropic/claude-haiku-4-5` | Conversation model (`provider/model`) |
| `WADDINGTON_LLM_BACKEND` | `anthropic` | The C-arm's own LLM backend: `anthropic` (benchmark) / `pi` (tool-less bridge, any provider) / `mock` |
| `WADDINGTON_PI_MODEL` | — | With `WADDINGTON_LLM_BACKEND=pi`: the provider/model the C-arm reasons on |

The benchmark path (`run_sequential.py`, direct `anthropic` backend, `claude-haiku`) is intentionally
**not** switched by these — deployment can use any provider; reproduction stays fixed.

---

## License

[MIT](LICENSE)
