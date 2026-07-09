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
conda env create -f experiments/environment.yml
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

> **Note**: tokens expire after ~8 hours. Re-run the command above if you get
> a 429 rate-limit error mid-experiment.

Experiments 01 (baselines) do **not** require a token.

## Step 3 — Reproduce experiments

### Option A: Run all experiments

```bash
bash experiments/run_all.sh          # ~8-12 hours total
```

### Option B: Run experiments individually

```bash
# Table 2, baseline rows (Random / Coreset / LOO-ML / OnlineAdaptive) — no API token
bash experiments/01_baselines.sh     # ~5-10 min

# Table 2, main result (A vs B vs C, 5 seeds) — requires API token
bash experiments/02_three_arm.sh     # ~2-3 hours

# Table 4, ablation study (4 ablations × 5 seeds) — requires API token
bash experiments/03_ablations.sh     # ~6-8 hours
```

## Output

All result files are written to `workspace/results/sequential/` as JSON.

| File | Content |
|---|---|
| `baselines.json` | Experiment 01 (Random, Coreset, StaticRanker, OnlineAdaptive) |
| `three_arm.json` | Experiment 02 (A=Coreset, B=LLM, C=Waddington) |
| `ablation_memory.json` | C vs C−memory |
| `ablation_llm.json` | C vs C−LLM |
| `ablation_ml.json` | C vs C−ML |
| `ablation_shuffled.json` | C vs shuffled gene names |

Each JSON maps `dataset → arm → [seed_0_result, ..., seed_4_result]`.

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

workspace/results/sequential/
└── *.json                  ← experiment outputs (written by run_sequential.py)
```

## Arm descriptions

| Arm | Flag | Description |
|---|---|---|
| Random | `random` | Uniform random selection |
| Coreset (A) | `coreset` | Greedy diversity-maximising k-center |
| LOO-LightGBM | `static_ranker` | Cross-experiment LightGBM prior, static |
| OnlineAdaptive | `online_adaptive` | PerTurboAgent: LOO prior + in-experiment retraining |
| LLM Reasoning (B) | `llm_reasoning` | Claude Haiku with LOO-ML fallback |
| **Waddington (C)** | `waddington_c` | **Online ML + LLM + DepMap routing (paper method)** |
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
