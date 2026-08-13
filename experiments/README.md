# Reproducing Paper Results

This directory contains all scripts needed to reproduce the experimental results
reported in the Waddington paper.

## Requirements

| Requirement | Details |
|---|---|
| Python | 3.12 (via conda) |
| GPU | Not required (CPU only) |
| RAM | ≥ 8 GB |
| Disk | ≥ 2 GB (datasets + feature files) |
| API token | Anthropic API key (for LLM experiments) |

## Step 1 — Create environment

```bash
conda env create -f environment.yml   # the canonical env file lives at the repo root
conda activate waddington-bio
```

## Step 2 — Configure API token

LLM-based experiments (B arm, C arm, ablations) call the Claude API via Anthropic.
Set your token once:

```bash
python3 experiments/setup_auth.py --token sk-ant-YOUR_TOKEN_HERE
```

To verify the token is valid at any time:

```bash
python3 experiments/setup_auth.py --check
```

> **Note**: `setup_auth.py` records an ~8-hour validity window. Re-run it when `--check` reports expiry.
> HTTP 429 means rate limiting, not token expiry; wait for the limit to clear rather than reauthorizing.

Experiments 01 (baselines) do **not** require a token.

## Step 3 — Reproduce experiments

### Option A: Run all experiments

```bash
bash experiments/run_all.sh          # long-running; total time depends on API throughput
```

### Option B: Run experiments individually

```bash
# Reported main table: all six methods × 20 seeds — requires API token
bash experiments/00_main_table_20seed.sh

# Five-seed companion baseline rows — no API token
bash experiments/01_baselines.sh     # ~5-10 min

# Supporting legacy/component comparison + honest 5-seed companion — requires API token
bash experiments/02_three_arm.sh     # ~2-3 hours

# Table 4, ablation study (4 ablations × 5 seeds) — requires API token
bash experiments/03_ablations.sh     # ~6-8 hours
```

The scripts recompute their outputs from scratch by default. Set `WADDINGTON_RESUME=1` only to continue
an interrupted rerun whose JSON already contains partial checkpoints; a fresh clone already contains
complete frozen outputs, so enabling resume on the first run would intentionally skip them.

## Output

Result files are written to `workspace/results/router/` (headline/router controls) and
`workspace/results/sequential/` (5-seed supporting analyses) as JSON.

| File | Content |
|---|---|
| `router/main_table_20seed.json` | Reported 20-seed main table (all six methods) |
| `router/legacy_waddington_20seed.json` | Matched 20-seed legacy target-aware control |
| `sequential/baselines.json` | Five-seed companion baselines |
| `sequential/three_arm.json` | Five-seed legacy A/B/C comparison used by supporting analyses |
| `router/clean_headline_w0.2_5seed.json` | Five-seed companion for the leakage-free C-arm |
| `sequential/ablation_memory.json` | C vs C−memory |
| `sequential/ablation_llm.json` | C vs C−LLM |
| `sequential/ablation_ml.json` | C vs C−ML |
| `sequential/ablation_shuffled.json` | C vs shuffled gene names |

Each JSON maps `dataset → arm → [seed_0_result, ...]`; the headline files contain 20 seeds and the
supporting files contain 5 unless their filename says otherwise.

## Code structure

```
waddington_select/
├── run_sequential.py       ← main entry point (all experiments go through here)
├── oracle.py               ← dataset truth-reveal interface
├── sequential_runner.py    ← per-arm evaluation loop
└── arms/
    ├── base.py             ← BaseArm interface
    ├── random_arm.py       ← Random baseline
    ├── coreset_arm.py      ← A arm: Coreset diversity selection
    ├── static_ranker_arm.py ← LOO LightGBM static prior
    ├── online_adaptive_arm.py ← PerTurboAgent-style online ML
    ├── llm_reasoning_arm.py  ← B arm: pure LLM reasoning
    ├── waddington_arm.py     ← memory utilities (shared by C arm)
    ├── waddington_c_arm.py ← C arm: Waddington (paper final)
    │
    ├── waddington_c_no_memory_arm.py   ← ablation: C − memory
    ├── waddington_c_no_llm_arm.py      ← ablation: C − LLM
    ├── waddington_c_no_ml_arm.py       ← ablation: C − ML (retrain)
    ├── waddington_c_shuffled_names_arm.py ← ablation: shuffled gene names
    │
    └── archive/            ← development-history arms (not needed for paper)
        └── waddington_v{2..13,15}_arm.py

workspace/evaluation/
├── lgbm_training_data.csv        ← feature matrix (v1: 9 PPI features)
├── lgbm_training_data_v2.csv     ← + 3 pan-cancer DepMap features
├── lgbm_training_data_v3.csv     ← + K562 Chronos (ACH-000551)
└── ...

workspace/data/bda_benchmark/
├── task_prompts/           ← per-dataset LLM task descriptions
└── ...                     ← ground-truth labels per dataset

workspace/results/
├── router/                 ← reported 20-seed table + matched router controls
└── sequential/             ← supporting 5-seed experiments and ablations
```

## Arm descriptions

| Arm | Flag | Description |
|---|---|---|
| Random | `random` | Uniform random selection |
| Coreset (A) | `coreset` | Greedy diversity-maximising k-center |
| LOO-LightGBM | `static_ranker` | Cross-experiment LightGBM prior, static |
| OnlineAdaptive | `online_adaptive` | PerTurboAgent: LOO prior + in-experiment retraining |
| LLM Reasoning (B) | `llm_reasoning` | Claude Haiku with LOO-ML fallback |
| **Waddington (C)** | `waddington_c` | **Online ML + LLM; global 0.8/0.2 fusion, metadata-selected DepMap policy, no anchor features (paper method)** |
| C − memory | `waddington_c_no_memory` | Ablation: no cross-experiment memory |
| C − LLM | `waddington_c_no_llm` | Ablation: online ML only |
| C − ML | `waddington_c_no_ml` | Ablation: static LOO prior + LLM |
| Shuffled names | `waddington_c_shuffled_names` | Ablation: anonymous gene identifiers |

## Custom runs

```bash
# Single dataset, 3 seeds
conda run -n waddington-bio python3 -m waddington_select \
    --arms waddington_c llm_reasoning \
    --datasets IFNG \
    --seeds 3

# Quick smoke test (1 dataset, 1 seed)
conda run -n waddington-bio python3 -m waddington_select \
    --arms waddington_c \
    --datasets Scharenberg22 \
    --seeds 1
```
