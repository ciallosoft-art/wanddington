# Waddington — compatibility notes

## What this repo is

A Python research project: a hybrid ML + LLM + verified-memory agent for sequential CRISPR gene
selection (the "C-arm"). The agent code is the `waddington_select`
package; data/results live under `workspace/`; the paper's tables are in `docs/`.

## How to run

Everything runs in the `waddington-bio` conda env. From the repo root:

```bash
# Reported 20-seed main table + matched legacy control
bash experiments/00_main_table_20seed.sh

# Five-seed supporting analyses
bash experiments/01_baselines.sh
bash experiments/02_three_arm.sh
bash experiments/03_ablations.sh

# Active GPT-5.6 Luna/none end-to-end experiment (strict checkpoint/resume)
bash experiments/04_gpt_luna_5repeat.sh

# Package directly (frozen reported main table protocol)
WADDINGTON_LLM_BACKEND=anthropic \
WADDINGTON_LLM_MODEL=claude-haiku-4-5-20251001 \
WADDINGTON_MEMORY_PATH=workspace/results/sequential/experience_memory.json \
WADDINGTON_FULL_SELECTED_CONTEXT=0 \
WADDINGTON_LLM_NAMED_ONLY=0 \
conda run -n waddington-bio python3 -m waddington_select \
  --arms random coreset static_ranker online_adaptive llm_reasoning waddington_c \
  --seeds 20 --rounds 5 --out workspace/results/router/main_table_20seed.json

# Active GPT memory must be conditioned on completed GPT calibration; use the staged script above.
```

LLM arms default to GPT-5.6 Luna with Codex reasoning `none` and need Codex OAuth.
Auth is self-contained in Waddington's own store
`~/.waddington/agent/auth.json`: either interactive OAuth via `node frontend/bin/waddington.js setup`
(select Codex). A raw Anthropic token is needed only for frozen historical reproduction.
The C-arm's LLM client (`llm_client.resolve_auth_json`) reads `~/.waddington` first, falling back to
`~/.feynman`; set `WADDINGTON_REUSE_FEYNMAN=1` to prefer feynman's store, or `WADDINGTON_AUTH_PATH` to override.

## Package structure

- `waddington_select/` is a proper Python package (relative imports; `pyproject.toml` at root).
- `run_sequential.py` is the experiment runner; `python -m waddington_select` is its entry point.
- `experiments/04_gpt_luna_5repeat.sh` is the canonical GPT workflow: calibration → strict
  GPT-calibrated memory → C-arm → validated merge. Do not resume the old `gpt_5_6_luna_minimal`
  pilot or mix it with this protocol.
- Data paths are resolved `REPO_ROOT`-relative (`REPO_ROOT = Path(__file__).resolve().parents[1]`
  for top-level modules, `parents[2]` for `arms/`). If you move the package, fix these.
- `arms/archive/` is frozen development history (waddington_v2..v15) — not used by the paper runs.
