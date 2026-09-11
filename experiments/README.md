# Experiments: Frozen Paper Runs and Active GPT Evaluation

Scripts 00–03 reproduce the frozen paper experiments. Scripts 04–05 are the active end-to-end GPT
evaluation; they write to a separate result tree and never relabel or overwrite the paper outputs.
Scripts 08–09 run BioDiscoveryAgent's other two difficulty tiers (see below).

## Requirements

| Requirement | Details |
|---|---|
| Python | 3.12 (via conda) |
| GPU | Not required (CPU only) |
| RAM | ≥ 8 GB |
| Disk | ≥ 2 GB (datasets + feature files) |
| Active GPT auth | ChatGPT/Codex OAuth in `~/.waddington/agent/auth.json` |
| Historical auth | Anthropic API key, only for frozen paper reproduction |

## Step 1 — Create environment

```bash
conda env create -f environment.yml   # the canonical env file lives at the repo root
conda activate waddington-bio
```

## Step 2 — Configure the active GPT backend

Authorize once, selecting **Codex (OpenAI / ChatGPT)**:

```bash
node frontend/bin/waddington.js setup
```

The current project default is `openai-codex/gpt-5.6-luna` with Codex reasoning effort `none`.
Run the five-replicate GPT experiment with:

```bash
bash experiments/04_gpt_luna_5repeat.sh
```

It runs in three provenance-separated stages: five-arm calibration (including GPT-B), fail-closed
memory generation from those GPT calibration results, then GPT-C evaluation and a validated merge.
GPT-B is explicitly treated as a composite because its unmatched slots use StaticRanker padding;
active GPT-C gives the LLM fusion vote only to genes GPT actually named. The protocol is
leave-one-phenotype-out and allows same-study sibling screens, which its manifest records.
The optional skill library is not used by this six-arm table and is not built unless
`WADDINGTON_BUILD_GPT_SKILLS=1`. Every arm replicate is atomically checkpointed; rerunning the same
command resumes only when the exact runtime/input fingerprint matches. Outputs are isolated under
`workspace/results/gpt/gpt_5_6_luna_none/` and carry manifest sidecars.

> Script 04 uses `--replicates 5 --seed-mode local`. Random receives a distinct deterministic local
> seed; Coreset/static/online are deterministic re-evaluations. Codex exposes no provider sampling
> seed, so GPT-B/C are repeated API invocations (independence not assumed), not reproducible model seeds.

## BDA difficulty tiers (scripts 08–09)

BioDiscoveryAgent grades the benchmark into three tiers. The paper's headline runs the middle one;
scripts 08 and 09 run the other two under exactly the same protocol — same batch sizes, same rounds,
same twenty evaluation replicates, same six arms, same task prompts — so the three are comparable.

| Tier | Candidates | Table | Script | Result dir |
|---|---|---|---|---|
| easier | 1 gene, full pool (essentials included) | `lgbm_training_data_tier_a{,_v3}.csv` | `08_tier_a_full_pool.sh` | `gpt/tier_a_full_pool_20rep/` |
| harder *(headline)* | 1 gene, CEGv2-filtered pool | `lgbm_training_data{,_v3}.csv` | `05_gpt_luna_20repeat.sh` | `gpt/gpt_5_6_luna_none_20rep/` |
| hardest | 2 genes, Horlbeck GI map | `lgbm_training_data_horlbeck_pairs{,_v3}.csv` | `09_tier_b_pairs.sh` | `gpt/tier_b_pairs_20rep/` |

Build the tables first (both builders self-verify against the frozen benchmark and refuse to write a
table that does not reproduce it):

```bash
conda run -n waddington-bio python3 workspace/evaluation/build_tier_a_unfiltered.py
conda run -n waddington-bio python3 workspace/evaluation/build_horlbeck_pairs.py   # needs tier A
bash experiments/08_tier_a_full_pool.sh
bash experiments/09_tier_b_pairs.sh        # reuses experiment 08's memory
conda run -n waddington-bio python3 workspace/evaluation/summarize_tiers.py
```

No arm code is tier-specific. A tier is selected entirely by `WADDINGTON_TRAINING_CSV`,
`WADDINGTON_TRAINING_CSV_V3` and `WADDINGTON_EXTRA_DATASETS`, which every consumer now reads from
`waddington_select/tables.py`. In the combinatorial tier a candidate is one opaque token
`"GENEA|GENEB"`, so the oracle, round loop, hit@R5 and result JSON are unchanged; only the LLM prompt
asks for `"GENE1 + GENE2"` and injects the 449-gene screen library, without which the model's
round-1 proposals are ~98% outside the measured pool.

## Historical Claude paper reproduction

Scripts 00–03 form the frozen paper suite. Scripts 00, 02, and 03 explicitly pin the historical
Claude model/backend and legacy prompt/memory paths; script 01 has no LLM. Set an Anthropic token only
if you need that historical reproduction:

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

## Run historical experiments

### Option A: Run all experiments

```bash
bash experiments/run_all.sh          # long-running; total time depends on API throughput
```

### Option B: Run experiments individually

```bash
# Reported main table: all six methods × 20 historical evaluation repeats — requires API token
bash experiments/00_main_table_20seed.sh

# Five-seed companion baseline rows — no API token
bash experiments/01_baselines.sh     # ~5-10 min

# Supporting legacy/component comparison + honest 5-seed companion — requires API token
bash experiments/02_three_arm.sh     # ~2-3 hours

# Table 4, ablation study (4 ablations × 5 historical evaluation repeats) — requires API token
bash experiments/03_ablations.sh     # ~6-8 hours

# Active end-to-end GPT evaluation (5 API replicates; local baselines are seeded)
bash experiments/04_gpt_luna_5repeat.sh

# BDA's other two difficulty tiers (build the tables first — see "BDA difficulty tiers")
bash experiments/08_tier_a_full_pool.sh    # 1-gene, full pool
bash experiments/09_tier_b_pairs.sh        # 2-gene combinations
```

Scripts 00–03 recompute their outputs from scratch by default. Set `WADDINGTON_RESUME=1` only to
continue a matching historical rerun. Script 04 always uses strict resume: every checkpoint must match
its manifest and runtime/input hashes, so an interrupted quota window can be continued safely.

## Output

Result files are written to `workspace/results/router/` (headline/router controls) and
`workspace/results/sequential/` (historical supporting analyses), while new GPT results go to
`workspace/results/gpt/`, as JSON.

| File | Content |
|---|---|
| `router/main_table_20seed.json` | Reported 20-seed main table (all six methods) |
| `router/legacy_waddington_20seed.json` | Matched 20-seed legacy target-aware control |
| `sequential/baselines.json` | Five-seed companion baselines |
| `sequential/three_arm.json` | Five-seed legacy A/B/C comparison used by supporting analyses |
| `router/clean_headline_w0.2_5seed.json` | Five-repeat companion for the target-label-free router/features C-arm |
| `sequential/ablation_memory.json` | C vs C−memory |
| `sequential/ablation_llm.json` | C vs C−LLM |
| `sequential/ablation_ml.json` | C vs C−ML |
| `sequential/ablation_shuffled.json` | C vs shuffled gene names |
| `gpt/gpt_5_6_luna_none/calibration_5rep.json` | Stage-1 five-arm GPT calibration checkpoints |
| `gpt/gpt_5_6_luna_none/experience_memory.json` | GPT-generated, fail-closed memory conditioned on Stage 1 |
| `gpt/gpt_5_6_luna_none/waddington_c_5rep.json` | Stage-3 GPT-C checkpoints |
| `gpt/gpt_5_6_luna_none/main_table_5rep.json` | Validated six-arm GPT table, created only after completion |
| `gpt/gpt_5_6_luna_none/main_table_5rep.manifest.json` | Component hashes and final GPT provenance |
| `gpt/gpt_5_6_luna_minimal/` | Superseded incomplete pilot; never resume or use as a headline |

Result JSON maps `dataset → arm → [result_0, ...]`. Frozen files retain the paper's historical
"seed" terminology. Active GPT files record `repeat_index`, any local seed, and
`provider_seed: null` for Codex-backed arms.

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
    ├── llm_reasoning_arm.py  ← B arm: GPT proposals + static LOO-ML padding
    ├── waddington_arm.py     ← memory utilities (shared by C arm)
    ├── waddington_c_arm.py ← C arm: active GPT protocol; paper scripts pin legacy vote semantics
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
├── lgbm_training_data_tier_a{,_v3}.csv        ← BDA "easier" tier: unfiltered full pool
├── lgbm_training_data_horlbeck_pairs{,_v3}.csv ← BDA "hardest" tier: 2-gene candidates
├── build_tier_a_unfiltered.py    ← builds the full-pool tables (verifies against the frozen ones)
├── build_horlbeck_pairs.py       ← builds the pair tables
├── summarize_tiers.py            ← side-by-side tier table (+ LaTeX)
└── ...

workspace/data/bda_benchmark/
├── task_prompts/           ← per-dataset LLM task descriptions
└── ...                     ← ground-truth labels per dataset

workspace/results/
├── router/                 ← reported 20-seed table + matched router controls
├── sequential/             ← supporting 5-seed experiments and ablations
└── gpt/                    ← active GPT calibration, memory, C-arm, and manifests
    ├── gpt_5_6_luna_none_20rep/    ← headline tier (1-gene, non-essential)
    ├── tier_a_full_pool_20rep/     ← BDA "easier" tier
    └── tier_b_pairs_20rep/         ← BDA "hardest" tier (2-gene)
```

## Arm descriptions

| Arm | Flag | Description |
|---|---|---|
| Random | `random` | Uniform random selection |
| Coreset (A) | `coreset` | Greedy diversity-maximising k-center |
| LOO-LightGBM | `static_ranker` | Cross-experiment LightGBM prior, static |
| OnlineAdaptive | `online_adaptive` | PerTurboAgent: LOO prior + in-experiment retraining |
| LLM Reasoning (B) | `llm_reasoning` | GPT-5.6 Luna/none proposals plus LOO-ML padding; composite, not pure LLM |
| **Waddington (C)** | `waddington_c` | **Online ML + GPT-named-only vote; global 0.8/0.2 weight, metadata DepMap policy, no anchor features** |
| C − memory | `waddington_c_no_memory` | Ablation: no cross-experiment memory |
| C − LLM | `waddington_c_no_llm` | Ablation: online ML only |
| C − ML | `waddington_c_no_ml` | Ablation: static LOO prior + LLM |
| Shuffled names | `waddington_c_shuffled_names` | Ablation: anonymous gene identifiers |

## Custom runs

```bash
# Single dataset, 3 repeated API evaluations (Codex has no provider seed)
conda run -n waddington-bio python3 -m waddington_select \
    --arms waddington_c llm_reasoning \
    --datasets IFNG \
    --replicates 3 --seed-mode local

# Quick smoke test (1 dataset, 1 replicate)
conda run -n waddington-bio python3 -m waddington_select \
    --arms waddington_c \
    --datasets Scharenberg22 \
    --replicates 1 --seed-mode local
```
