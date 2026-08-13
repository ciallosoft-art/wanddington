# Waddington — agent instructions

This repository is a hybrid ML + LLM + verified-memory agent that recommends which
genes to perturb next in a CRISPR screen, for a given phenotype. When a user (an experimenter)
asks you to select / recommend / rank genes for a phenotype, drive the tools below — do not answer
from your own parametric knowledge alone.

Everything runs in the `waddington-bio` conda env, from the repo root.

## Canonical conversational entry (for scientists) — `frontend/`

The primary way a scientist uses this project is the tool-less conversational shell:

```bash
cd frontend && node bin/waddington.js
```

On launch it authorizes a model provider (Claude / Codex / Gemini) in Waddington's own token store
then offers **Terminal (CLI)** or **Web UI**. The scientist chats in natural language; the shell's
LLM only *routes intent + narrates* (it never picks genes), and the deterministic **C-arm pipeline**
(`waddington_select.suggest` / `.simulate`) does the selection. Provider is switchable with
`--model provider/model`. This is built on **pi-ai** (the tool-less LLM layer), not feynman-the-agent.

The `suggest` / `simulate` commands below are the same brain that entry drives; use them directly
when you (an agent) are operating the repo. **Do not** use the free tool-using
`skills/gene-selection-agent` agent for real selection — it is benchmark-only and loses to the
pipeline (0.209 vs 0.256 in the matched legacy 5-seed diagnostic).

## Primary action — recommend genes to perturb

```bash
conda run -n waddington-bio python3 -m waddington_select.suggest --dataset <PHENOTYPE> \
    [--n N] [--tested-hits GENE ...] [--tested-misses GENE ...] [--exclude GENE ...] [--skills]
```

- `--dataset` must be one of the nine supported phenotypes (see below).
- `--n` = how many genes to recommend (default = the screen's batch size).
- **`--tested-hits`** = genes already tested that WERE hits; **`--tested-misses`** = genes tested
  that were NOT hits. These are experimental feedback: they retrain the online ML model and are
  fed to the LLM as history, so the next recommendation adapts (true sequential selection). Use
  them whenever the experimenter reports results from a prior round.
- `--exclude` = genes to leave out of recommendations when the outcome is unknown/irrelevant.
- `--skills` = use the evolving skill library instead of flat cross-experiment memory.

The command prints a ranked list of gene symbols (and notes any supplied genes not in this
phenotype's pool). With no `--tested-*`, it gives the cold-start first-round recommendation; with
feedback, it gives the adapted next round. Relay the genes and briefly explain the phenotype
context. This is a *forward* recommendation (no ground-truth oracle involved).

**Supported phenotypes** (`--dataset` values): `IFNG`, `IL2`, `Sanchez21`, `Sanchez21_down`,
`Carnevale22`, `Scharenberg22`, `Steinhart`, `Replogle_K562_essential`, `Replogle_K562_gwps`.
A brand-new phenotype is not yet supported (its ML features must be precomputed first) — say so
rather than guessing.

## Show how a whole sequential experiment would go — `simulate`

When a user wants to *see* how Waddington would drive a multi-round CRISPR campaign for one of the
phenotypes (a demo of "what using this on my experiment looks like"), run:

```bash
conda run -n waddington-bio python3 -m waddington_select.simulate --dataset <PHENOTYPE> [--rounds 5] [--skills]
```

It replays a full sequential campaign round by round — each round the C-arm recommends a batch, the
benchmark's ground-truth oracle stands in for the wet-lab result, and the next round adapts. It
prints per-round hits, cumulative discovery, and what it would test next. Relay the trajectory and
the final hit ratio. (For a real experiment with the user's own results instead of the oracle,
use `suggest` with `--tested-hits/--tested-misses` iteratively.)

## Choosing the LLM provider

The gene-selection pipeline's own reasoning model defaults to Claude. To run it through the frontend's
tool-less, multi-provider pi-ai bridge (for example with Codex), set an env var — no code change:

```bash
WADDINGTON_LLM_BACKEND=pi conda run -n waddington-bio python3 -m waddington_select.suggest --dataset IFNG
```

`pi` shells out to `node frontend/bin/waddington.js complete`, a pure completion bridge with OAuth
auto-refresh and no tools. `WADDINGTON_PI_MODEL="provider/model"` picks a specific model. Default
(`anthropic`) is faster and is what the paper benchmark uses. `WADDINGTON_REUSE_FEYNMAN=1` is only an
opt-in way to reuse an existing feynman token store.

## Benchmark / evaluation (for reproduction, not for recommending)

```bash
bash experiments/00_main_table_20seed.sh  # reported six-method table + matched legacy control
bash experiments/01_baselines.sh   # baselines (no API)
bash experiments/02_three_arm.sh   # supporting 5-seed legacy/component comparison
bash experiments/03_ablations.sh   # four legacy 5-seed ablations
```

Rebuild memory / skills (needs API):

```bash
conda run -n waddington-bio python3 -m waddington_select.memory_builder     # experience_memory.json
conda run -n waddington-bio python3 -m waddington_select.skill_builder        # skill_library.json
```

## How the recommendation is produced (for explaining to the user)

`waddington_c` fuses three signals with a globally fixed 0.8/0.2 ML/LLM weight
(`arms/waddington_c_arm.py`): online-retrained LightGBM over PPI/KEGG/pLI/metadata-selected DepMap
features + Claude/LLM biological reasoning + cross-experiment memory (or the skill library). The
per-phenotype target-aware router exists only behind `WADDINGTON_LEGACY_ROUTER=1`. See `README.md` and
`docs/` for details.

## Conventions

- Never fabricate benchmark numbers — run the command and report what it prints.
- If a command needs provider auth and it is missing/expired, tell the user to run
  `node frontend/bin/waddington.js setup` (OAuth) or
  `python experiments/setup_auth.py --token …` (direct Anthropic benchmark path).
