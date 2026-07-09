# Waddington — agent instructions

This repository is **System 2**: a hybrid ML + LLM + verified-memory agent that recommends which
genes to perturb next in a CRISPR screen, for a given phenotype. When a user (an experimenter)
asks you to select / recommend / rank genes for a phenotype, drive the tools below — do not answer
from your own parametric knowledge alone.

Everything runs in the `waddington-bio` conda env, from the repo root.

## Primary action — recommend genes to perturb

```bash
conda run -n waddington-bio python3 -m waddington_select.suggest --dataset <PHENOTYPE> [--n N] [--exclude GENE ...] [--skills]
```

- `--dataset` must be one of the nine supported phenotypes (see below).
- `--n` = how many genes to recommend (default = the screen's batch size).
- `--exclude` = genes the experimenter has already tested (won't be recommended again).
- `--skills` = use the evolving skill library instead of flat cross-experiment memory.

The command prints a ranked list of gene symbols. Relay them, and briefly explain the phenotype
context if useful. This is a *forward* recommendation (no ground-truth oracle involved).

**Supported phenotypes** (`--dataset` values): `IFNG`, `IL2`, `Sanchez21`, `Sanchez21_down`,
`Carnevale22`, `Scharenberg22`, `Steinhart`, `Replogle_K562_essential`, `Replogle_K562_gwps`.
A brand-new phenotype is not yet supported (its ML features must be precomputed first) — say so
rather than guessing.

## Choosing the LLM provider

The gene-selection agent's own reasoning model defaults to Claude. To run it on another provider
that feynman is authenticated for (e.g. codex), set an env var — no code change:

```bash
WADDINGTON_LLM_BACKEND=pi conda run -n waddington-bio python3 -m waddington_select.suggest --dataset IFNG
```

`pi` shells out to `feynman --prompt`, reusing feynman's provider routing + OAuth. `WADDINGTON_PI_MODEL="provider/model"` picks a specific model. Default (`anthropic`) is faster and is what the paper benchmark uses.

## Benchmark / evaluation (for reproduction, not for recommending)

```bash
conda run -n waddington-bio python3 -m waddington_select --arms waddington_c waddington_c_skills --seeds 5 --rounds 5
bash experiments/01_baselines.sh   # baselines (no API)
bash experiments/02_three_arm.sh   # A: coreset / B: llm_reasoning / C: waddington_c
bash experiments/03_ablations.sh   # four ablations
```

Rebuild memory / skills (needs API):

```bash
conda run -n waddington-bio python3 -m waddington_select.memory_builder     # experience_memory.json
conda run -n waddington-bio python3 -m waddington_select.skill_builder        # skill_library.json
```

## How the recommendation is produced (for explaining to the user)

`waddington_c` fuses three signals, routed per phenotype (`arms/waddington_c_arm.py`):
online-retrained LightGBM over PPI/KEGG/pLI/DepMap features + Claude/LLM biological reasoning +
cross-experiment memory (or the skill library). See `README.md` and `docs/` for details.

## Conventions

- Never fabricate benchmark numbers — run the command and report what it prints.
- If a command needs an Anthropic/feynman token and it is missing/expired, tell the user to run
  `feynman model login` (or `python experiments/setup_auth.py --token …` for the anthropic path).
