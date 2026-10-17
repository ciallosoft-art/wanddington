# Waddington

**A hybrid ML + LLM + verified-memory agent for sequential CRISPR gene selection.**

Named after Conrad Waddington (1905–1975) and his epigenetic landscape. Given a target phenotype
and a genome-scale pool of candidate genes, the agent picks — over a few rounds of CRISPR-screen
feedback — which genes to perturb next, aiming to find "hit" genes as fast as possible.

---

## Pipeline (the C-arm)

`waddington_select` runs a **5-round sequential loop**: each round it selects a batch of genes, an
oracle (benchmark) or the wet lab (deployment) reveals hit / non-hit, and that feedback conditions
the next round. The selector (`waddington_c`) fuses three signals with a **globally fixed
0.80 ML / 0.20 LLM weight** (`arms/waddington_c_arm.py`):

| Component | What it does |
|-----------|--------------|
| **Online adaptive ML** (`OnlineAdaptiveArm`, LightGBM) | Scores every gene from PPI / KEGG / pLI / DepMap features; **retrains each round** on the revealed feedback. |
| **LLM reasoning** (`LLMReasoningArm`, GPT-5.6 Luna `none`) | Names candidate genes from parametric biology, conditioned on the task and the revealed history. Only genes the model actually names receive the fusion vote. |
| **Cross-experiment memory** (`memory_builder`) | Strategy summaries distilled from the *other* phenotypes, injected into the LLM prompt through a verified admission gate. Leave-one-phenotype-out: the target phenotype is excluded. |

Round 1 is the ML prior alone (no in-experiment data yet); from round 2 the ML retrains on the
revealed `(gene, label)` pairs and re-ranks the pool. Any batch shortfall from the LLM is padded
from the static ranker (padding gets no vote). A legacy per-phenotype router that reads the target
screen's *realized* hit rate exists only behind `WADDINGTON_LEGACY_ROUTER=1` — it is **not** the
reported system. Details and results: [`docs/results_tables.tex`](docs/results_tables.tex).

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
├── memory_builder.py     #   cross-experiment memory w/ verified admission
└── arms/                 #   selection arms (C-arm, baselines, ablations)

frontend/                 # the conversational entry (CLI + Web, Node) — see frontend/README.md
├── bin/waddington.js     #   `node bin/waddington.js` → auth → CLI/Web → chat
└── src/                  #   tool-less pi-ai bridge + intent routing; drives suggest/simulate

workspace/
├── evaluation/           # feature pipeline + LOO LightGBM training data
├── models/               # trained per-dataset LightGBM models
├── data/bda_benchmark/   # task prompts + benchmark labels
└── results/              # frozen paper runs (sequential/, router/) + GPT experiments (gpt/)

docs/                     # results_tables.tex (paper) + research iteration history
experiments/              # reproduction scripts 00–03 (frozen) + active GPT experiment 04
```

---

## Setup (fresh clone)

Two halves: a Python **brain** and a Node **frontend**. Frozen features / caches / prompts / models
are committed; active GPT memory is built by experiment 04.

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

No API key needs to be pre-configured: on first launch the frontend prompts you to authorize Codex
through ChatGPT OAuth (stored at `~/.waddington/agent/auth.json`). Claude and Gemini remain
available only as compatibility providers; neither is an implicit fallback.

---

## Use it (scientists)

The primary way to use the agent is natural-language chat that drives the C-arm pipeline (the shell's
LLM only routes intent and narrates — it never picks genes):

```bash
cd frontend && node bin/waddington.js      # authorize → pick Terminal (CLI) or Web UI → chat
```

See [`frontend/README.md`](frontend/README.md). Or drive the brain directly:

```bash
# forward recommendation (real experiment; feed prior-round results back to adapt the next round)
conda run -n waddington-bio python3 -m waddington_select.suggest --dataset IFNG \
    --tested-hits STAT1 JAK2 --tested-misses ACTB GAPDH

# narrated demo campaign (the benchmark oracle plays the wet lab)
conda run -n waddington-bio python3 -m waddington_select.simulate --dataset IFNG --rounds 5
```

Supported phenotypes (`--dataset`): `IFNG`, `IL2`, `Sanchez21`, `Sanchez21_down`, `Carnevale22`,
`Scharenberg22`, `Steinhart`, `Replogle_K562_essential`, `Replogle_K562_gwps`, plus any phenotype you
register (the chat's onboarding does this for you):

```bash
conda run -n waddington-bio python3 -m waddington_select.phenotype register --name MyScreen \
    --genes-file my_library.csv \
    --task "genes that regulate IL-2 secretion" --measurement "IL-2 FACS" [--tier non_essential|full_pool] \
    --modality CRISPRi --cell-line "primary T cells" \
    --hit-rule threshold --threshold 1.5        # or: --hit-rule fdr --max-fdr 0.05 / --hit-rule label

# each round: derive hits for the tested batch from that round's readout, by the registered rule
conda run -n waddington-bio python3 -m waddington_select.ingest --dataset MyScreen \
    --file round1_results.csv --genes STAT1 JAK1 ... --json
```

**Hit rule.** How a tested gene counts as a hit is registered with the phenotype and fixed for the whole
campaign, so it is decided before any result is seen: `label` (your readouts carry a hit column),
`threshold` (`|score| >= X`, or `--direction up|down` for one side), `fdr` (`fdr < q`; MAGeCK's
`pos|fdr` / `neg|fdr` are recognised) or `top_fraction` (the top x% of the screen). A top-x% call ranks a
gene against the *whole* screen, so it only works when the readout covers the screen (replaying a completed
screen); `ingest` refuses it, and its legacy top-5% default, on a readout covering less than 90% of the
candidate pool, i.e. one round's batch, where it would call ~5% of every batch a hit whatever the effects.
For one batch per round the screen-wide hit count is unknown, so progress is reported as hits per tested
gene rather than hit@R5.

**Design metadata.** `--modality KO|CRISPRi|CRISPRa|unknown`, `--cell-line` and `--curated-essential`
(the library is a curated essential-gene panel) are known before the screen is run, and set which DepMap
features the prior uses by the same rule as the benchmarks (`router_protocol.feature_policy_from`): none for
CRISPRa (DepMap is knockout-derived, and a gene dispensable when deleted can be potent when over-expressed)
or a curated essential panel (an essentiality prior would restate the selection), pan-cancer only in K562
(the K562 feature would restate an essentiality label on its own cell line), otherwise pan-cancer + K562.
Leaving the modality out records it as `unknown`, treated as loss-of-function. Anchor genes (`--anchors`)
are optional: the reported C-arm drops the anchor-relative features, so only the legacy configuration
(`WADDINGTON_LEGACY_ROUTER=1`) uses them; without anchors those features are 0.

**Single-gene tier.** `non_essential` (default) is BDA's harder tier and the paper's headline: the 684
CEGv2 core-essential genes (`waddington_select/data/CEGv2.txt`) are removed from the candidate pool, so
the budget goes to phenotype-specific genes rather than generic essentials. `full_pool` is BDA's easier
tier: they stay in the pool, and the prior is trained on the full-pool tables, the only ones that contain
those genes. The two-gene (combinatorial) tier is not offered: on it the reported system loses to
BioDiscoveryAgent (paper, combinatorial tier).

A recommendation reads the cross-experiment memory of the reported run for its tier (the headline run's,
or the full-pool run's) unless `WADDINGTON_MEMORY_PATH` is set.

---

## Reproduce the benchmark

Two tracks — the active **GPT** protocol (the reported primary system) and the frozen **Claude**
paper runs — write to separate directories and never overwrite each other.

**GPT (primary):**

```bash
node frontend/bin/waddington.js setup       # authorize Codex (OpenAI / ChatGPT), once
bash experiments/04_gpt_luna_5repeat.sh     # calibrate → build verified memory → evaluate GPT-C
```

Outputs land in `workspace/results/gpt/gpt_5_6_luna_none/`. The script checkpoints each arm-repeat
atomically and resumes only when model, config, memory, prompts and feature-table hashes all match.

**Frozen Claude paper table (reproduction):**

```bash
bash experiments/00_main_table_20seed.sh    # 6-method main table + matched legacy control, 20 seeds
bash experiments/01_baselines.sh            # baselines, no API needed
bash experiments/02_three_arm.sh            # component comparison (5-seed)
bash experiments/03_ablations.sh            # component ablations (5-seed)
```

Or drive the runner directly (GPT default; the frozen scripts pin the `anthropic`/Claude backend
themselves):

```bash
conda run -n waddington-bio python3 -m waddington_select \
    --arms random coreset static_ranker online_adaptive llm_reasoning waddington_c \
    --seeds 20 --rounds 5
```

Arm names: `random`, `coreset`, `static_ranker`, `online_adaptive`, `llm_reasoning`, `waddington_c`,
plus the ablations `waddington_c_no_memory` / `_no_llm` / `_no_ml` / `_shuffled_names`.

> **Replicate caveat.** Only Random consumes a local RNG seed; Coreset and the ML arms are
> deterministic re-evaluations. The Codex OAuth endpoint exposes no sampling seed, so GPT-B and GPT-C
> are repeated API invocations (independence not assumed), not reproducible LLM seeds.

---

## Configuration

Environment variables (all optional; sensible defaults):

| Variable | Default | Purpose |
|----------|---------|---------|
| `WADDINGTON_AUTH_PATH` | `~/.waddington/agent/auth.json` (`WADDINGTON_REUSE_FEYNMAN=1` reuses `~/.feynman`) | OAuth/API token store used by the frontend |
| `WADDINGTON_PY` | `conda run -n waddington-bio python3` | How the frontend invokes the Python brain |
| `WADDINGTON_CHAT_MODEL` | `openai-codex/gpt-5.6-luna` | Conversation model (`provider/model`) |
| `WADDINGTON_LLM_BACKEND` | `pi` | C-arm backend: `pi` (GPT/Codex default) / `anthropic` (legacy reproduction) / `mock` |
| `WADDINGTON_PI_MODEL` | `openai-codex/gpt-5.6-luna` | Provider/model used by the C-arm |
| `WADDINGTON_PI_REASONING` | `none` | Codex reasoning effort |
| `WADDINGTON_MEMORY_PATH` | benchmark runs: `workspace/results/gpt/gpt_5_6_luna_none/experience_memory.json` (also where `memory_builder` writes); `suggest` / the frontend: the reported run's memory for the phenotype's tier | Cross-experiment episodic memory |
| `WADDINGTON_LEGACY_ROUTER` | `0` | `1` enables the legacy target-aware router (reads the realized hit rate; not the reported system) |
| `WADDINGTON_LEGACY_TASK_PROMPTS` | `0` | `1` restores the historical Replogle fitness wording for frozen reproduction; active prompts describe the actual Perturb-seq transcriptomic readouts |
| `WADDINGTON_LLM_NAMED_ONLY` | `1` | Fusion bonus only to model-named genes; frozen Claude scripts pin `0` to reproduce the historical padded vote |

Active deployment and new experiments default to GPT; the historical reproduction scripts pin the
`anthropic` backend, Claude model and frozen memory paths, and neither changes the provenance of the
other.

The Replogle task correction changes the prompt protocol. Active memory and optional skills must
be rebuilt from calibration under the corrected prompts; older artifacts are skipped with a warning.
The frozen scripts 00/02/03 explicitly retain the historical wording. Existing result files are
unchanged, and strict resume rejects older protocols. To start a fresh active calibration without
overwriting an old run, choose a new directory, for example:

```bash
WADDINGTON_GPT_RESULT_DIR=workspace/results/gpt/replogle_transcriptomics_v2 \
  bash experiments/04_gpt_luna_5repeat.sh
```

Offline regression checks (no provider calls):

```bash
conda run -n waddington-bio python -m unittest discover -s tests -v
npm --prefix frontend test
```

---

## License

[MIT](LICENSE)
