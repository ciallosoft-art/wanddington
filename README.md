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
| **LLM reasoning** (`LLMReasoningArm`, GPT-5.6 Luna/`none` by default) | Names genes from parametric biology, conditioned on task and feedback; when nested in C it also receives cross-experiment memory. Standalone B has no memory and pads unmatched slots with the LOO ranker, so it is a GPT+ML composite rather than a pure-LLM intervention. |
| **Cross-experiment memory** (`memory_builder`) | Strategy summaries distilled from the *other* phenotypes, injected into the LLM prompt. Active memory is conditioned on the completed GPT calibration table and admitted only when both mechanical and LLM-verifier gates pass; malformed/failed verification is rejected. Exact target entries are excluded, while the formal first GPT run retains same-study siblings and declares this phenotype-LOO scope in its manifest. |

**Fusion.** The reported system uses one global ML 0.80 / LLM 0.20 weight. The active GPT protocol
retains that routing/weight but corrects an attribution bug in the historical harness: only genes GPT
actually names receive the LLM vote; StaticRanker padding does not. Frozen Claude scripts explicitly
restore the historical padded-vote behavior.

> **Legacy target-aware router (analyzed, not reported).** An earlier variant
> (`waddington_c_arm._classify`) picks the fusion mode from the target screen's *realized* hit rate —
> `ml_heavy` (ML 0.80 / LLM 0.20), `baseline` (0.60 / 0.40), or `two_stage` (ML shortlists 384, LLM
> re-ranks). Because that reads a test-set statistic, it is a **leakage** variant kept only for the
> §honest-router analysis (avg 0.259 over the matched 20-seed run, with no detected difference from the
> reported 0.253). It is **not** the system we report. Enable it with
> `WADDINGTON_LEGACY_ROUTER=1`; the reported router/feature settings remain the defaults, while
> frozen scripts pin the historical padded-vote policy explicitly.

---

## Published results (historical Claude backend)

These frozen numbers were produced with Claude Haiku and remain unchanged for paper
reproducibility. The paper calls them 20 seeds; implementation audit shows they are 20 evaluation
repeats with fixed local arm seeds, while only LLM-backed arms make API invocations, across nine datasets:

| Method | avg hit@R5 |
|--------|-----------|
| Random | 0.066 |
| A: Coreset | 0.134 |
| LOO-LightGBM (static) | 0.217 |
| OnlineAdaptive | 0.224 |
| B: LLM reasoning | 0.222 |
| **C: Waddington (ours, target-label-free router/features)** | **0.253** |

The reported system is the **target-label-free router/feature** configuration (0.253): a *global* fusion
weight (`w_llm = 0.2`, chosen on non-target screens — never the target's realized hit rate), a
metadata-only DepMap feature policy, and **no** anchor-relative features. An earlier **target-aware
routed** variant scores 0.259 over a matched 20-seed run, but it reads the target screen's realized hit rate to pick its fusion
mode (a test-set statistic), so we do **not** report it as the system — the two are statistically
indistinguishable (paired −0.006, 95% CI [−0.019, 0.007]). See `§honest-router` in the paper.

> **Where the advantage comes from (be honest):** the dominant ingredient is the **cross-experiment
> supervised prior**, not agentic reasoning. A static LOO-LightGBM with *no* LLM already reaches 0.187
> on the six screens shared with BioDiscoveryAgent — nearly double BDA's LLM agent (0.105) and
> GeneDisco's active learning (0.113). Online adaptation and constrained LLM reasoning add a further,
> smaller gain (against its *matched* static prior, 0.231 → 0.253, +0.022, 95% CI [0.012, 0.036]).
> On genuinely *novel* (non-recurrent) hits the full system does **not** reliably beat the prior.

> **Frozen implementation erratum:** the paper's original fusion equation said only model-named genes
> received the LLM bonus, but the historical C code gave that bonus to the whole LLM-branch batch,
> including StaticRanker padding; attribution counts did exclude padding. The active GPT protocol fixes
> this with a named-only vote. Claude-vs-GPT scores are therefore not a one-variable provider comparison.

Full per-dataset tables, the ablation study (memory / LLM / online-ML / gene-name-shuffle), the
target-label-free router analysis, and the architecture progression are in
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
└── results/
    ├── sequential/       # frozen Claude supporting runs + memory
    ├── router/           # frozen paper headline/router controls
    └── gpt/              # GPT experiments, memory and provenance manifests

docs/                     # research iteration history (v1..v26) + results_tables.tex
experiments/              # frozen reproduction scripts 00–03 + active GPT experiment 04
```

---

## Setup (fresh clone)

Two halves: a Python **brain** and a Node **frontend**. Frozen paper features/caches/prompts/models are
committed; active GPT memory is built by experiment 04 after calibration.

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

No API key needs to be pre-configured: on first launch the frontend prompts you to authorize
Codex through ChatGPT OAuth. Claude and Gemini remain available only as compatibility providers.

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

The active default is `openai-codex/gpt-5.6-luna` with reasoning effort `none`, for both the
conversation shell and the Python C-arm. It uses Waddington's OAuth store at
`~/.waddington/agent/auth.json`; no provider is selected as an implicit fallback.

## End-to-end GPT experiment

The formal GPT evaluation keeps the published target-label-free routing, 0.8/0.2 weight and feature
policy, replaces the LLM-dependent stack without importing Claude-era performance summaries, and
uses the corrected **GPT-named-only** fusion vote. It is therefore a new audited GPT protocol, not a
claim that the frozen Claude number was reproduced by changing one provider string. It writes to a
separate directory and never overwrites the paper outputs:

```bash
node frontend/bin/waddington.js setup  # choose Codex (OpenAI / ChatGPT), once
bash experiments/04_gpt_luna_5repeat.sh
```

The script has three stages: (1) calibrate Random/Coreset/static ML/online ML/GPT-B+ML-padding on all nine
screens; (2) build strictly verified GPT memory from that calibration; (3) evaluate GPT-C with that
memory. It checkpoints every arm-repeat atomically. Resume is accepted only when model, effective
configuration, memory, source, bridge, package-lock, task prompts, and feature-table hashes match.
Each result also records the final selected genes, all valid GPT proposals, the nested StaticRanker
padding proposals, which GPT proposals survived final C fusion, and revealed hit genes by round, so a
concrete hit-gene decision can be audited without replaying API calls.
Completed outputs are created under `workspace/results/gpt/gpt_5_6_luna_none/`.

> **Replicate caveat.** In the formal six-arm set, only Random consumes a local RNG seed; Coreset and
> the ML arms are deterministic re-evaluations. The ChatGPT/Codex OAuth endpoint exposes no model
> sampling seed, so GPT-B and GPT-C are five repeated API invocations, with independence not assumed—
> not five reproducible LLM seeds. Manifests record `provider_seed: null`. The earlier
> interrupted `gpt_5_6_luna_minimal/` checkpoint is retained as a pilot only: it used Claude-era
> performance-conditioned memory and must not be merged with the formal run.

## Reproduce the historical paper benchmark

The historical benchmark is kept separate from the active GPT deployment and frozen on
`claude-haiku` for reproducibility.
The scripts also clear experimental routing/feature overrides before launching; the equivalent direct
commands below assume those overrides are not set in the parent shell.

```bash
bash experiments/00_main_table_20seed.sh  # reported main table + matched legacy, both at 20 seeds
bash experiments/01_baselines.sh          # 5-seed companion baselines; no API needed
bash experiments/02_three_arm.sh          # legacy/component run + 5-seed honest companion
bash experiments/03_ablations.sh          # legacy 5-seed component ablations

# Or run the runner directly (from repo root)
WADDINGTON_LLM_BACKEND=anthropic WADDINGTON_LLM_MODEL=claude-haiku-4-5-20251001 \
WADDINGTON_MEMORY_PATH=workspace/results/sequential/experience_memory.json \
WADDINGTON_FULL_SELECTED_CONTEXT=0 WADDINGTON_LLM_NAMED_ONLY=0 \
conda run -n waddington-bio python3 -m waddington_select \
    --arms random coreset static_ranker online_adaptive llm_reasoning waddington_c \
    --seeds 20 --rounds 5 --out workspace/results/router/main_table_20seed.json

# Optional audited reconstruction from frozen performance summaries (not the byte-identical paper memory)
WADDINGTON_LLM_BACKEND=anthropic WADDINGTON_LLM_MODEL=claude-haiku-4-5-20251001 \
conda run -n waddington-bio python3 -m waddington_select.memory_builder \
  --use-legacy-performance --out workspace/results/sequential/experience_memory_rebuilt_audit.json
```

**Reported routing vs active provider.** The default `waddington_c` routing/features match the
reported target-label-free router/feature method: global `w_llm = 0.2`, metadata feature policy, no anchor features —
none reads target outcomes. The value 0.253 belongs only to the frozen Claude run; the active GPT
default must be evaluated separately:

```bash
# Reproduce reported target-label-free router/feature C (Claude, paper protocol)
WADDINGTON_LLM_BACKEND=anthropic WADDINGTON_LLM_MODEL=claude-haiku-4-5-20251001 \
WADDINGTON_MEMORY_PATH=workspace/results/sequential/experience_memory.json \
WADDINGTON_FULL_SELECTED_CONTEXT=0 WADDINGTON_LLM_NAMED_ONLY=0 \
conda run -n waddington-bio python3 -m waddington_select \
    --arms waddington_c --seeds 20 --rounds 5

# Final-system paired statistics (matched-LOO, vs baselines, novel-hit)
conda run -n waddington-bio python3 -m waddington_select.analysis.final_system_stats
```

To reproduce the **legacy target-aware routed** configuration (avg 0.259 over 20 seeds, reads the
realized hit rate), set one flag. The detailed legacy SHAP / decomposition / ablation analyses remain
paired 5-seed runs with an average near 0.256:

```bash
WADDINGTON_LLM_BACKEND=anthropic WADDINGTON_LLM_MODEL=claude-haiku-4-5-20251001 \
  WADDINGTON_MEMORY_PATH=workspace/results/sequential/experience_memory.json \
  WADDINGTON_FULL_SELECTED_CONTEXT=0 WADDINGTON_LLM_NAMED_ONLY=0 WADDINGTON_LEGACY_ROUTER=1 \
  conda run -n waddington-bio python3 -m waddington_select \
    --arms waddington_c --seeds 20 --rounds 5 \
    --out workspace/results/router/legacy_waddington_20seed.json
```

(`experiments/03_ablations.sh` sets `WADDINGTON_LEGACY_ROUTER=1` for you; `02_three_arm.sh` produces
the paired 5-repeat legacy and target-label-free-router supporting runs.)

Arm names: `random`, `coreset`, `static_ranker`, `online_adaptive`, `llm_reasoning`,
`waddington_c`, and the ablations `waddington_c_no_memory` / `_no_llm` / `_no_ml` /
`_shuffled_names`.

## Configuration

Environment variables (all optional; sensible defaults):

| Variable | Default | Purpose |
|----------|---------|---------|
| `WADDINGTON_AUTH_PATH` | `~/.waddington/agent/auth.json` (own store; `WADDINGTON_REUSE_FEYNMAN=1` reuses `~/.feynman`) | OAuth/API token store used by the frontend |
| `WADDINGTON_PY` | `conda run -n waddington-bio python3` | How the frontend invokes the Python brain |
| `WADDINGTON_CHAT_MODEL` | `openai-codex/gpt-5.6-luna` | Conversation model (`provider/model`) |
| `WADDINGTON_LLM_BACKEND` | `pi` | The C-arm backend: `pi` (GPT/Codex default) / `anthropic` (legacy reproduction) / `mock` |
| `WADDINGTON_PI_MODEL` | `openai-codex/gpt-5.6-luna` | Provider/model used by the C-arm |
| `WADDINGTON_PI_REASONING` | `none` | Codex reasoning effort; injected explicitly into the OAuth Responses payload |
| `WADDINGTON_MEMORY_PATH` | `workspace/results/gpt/gpt_5_6_luna_none/experience_memory.json` | Active GPT-calibrated episodic memory |
| `WADDINGTON_SKILL_LIBRARY_PATH` | `workspace/results/gpt/gpt_5_6_luna_none/skill_library.json` | Optional GPT skill artifact; not used by the six-arm main table |
| `WADDINGTON_FULL_SELECTED_CONTEXT` | `1` | Include the full tested set in GPT prompts; frozen Claude scripts pin the historical truncated context |
| `WADDINGTON_LLM_NAMED_ONLY` | `1` | Give the fusion bonus only to model-named genes; frozen Claude scripts pin `0` to reproduce the historical padded vote |

The historical reproduction scripts explicitly pin the direct `anthropic` backend, Claude model,
and frozen memory paths. Active deployment and new experiments default to GPT; neither changes the
provenance of the reported Claude runs.

---

## License

[MIT](LICENSE)
