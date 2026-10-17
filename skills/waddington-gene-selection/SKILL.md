---
name: waddington-gene-selection
description: Recommend which genes to perturb next in a sequential, multi-round single-gene CRISPR screen (KO, CRISPRi or CRISPRa) with the Waddington pipeline; register a scientist's own screen library; turn each round's readout into hits and misses for the next round. Use when a user asks which genes to screen, test or perturb next, wants to onboard their own CRISPR screen, or reports results from a round. Not for two-gene (combinatorial) screens.
---

# Waddington: next-round gene selection for a CRISPR screen

Waddington ranks a screen's candidate genes with a LightGBM model trained across nine public screens. The
model is retrained each round on the results so far, and its ranking is fused with an LLM's named picks.
Your job is to run this loop for the scientist and relay its output:

1. Register the screen once.
2. Each round, ask `suggest` for a batch.
3. The scientist tests the batch.
4. `ingest` turns the readout into hits and misses.
5. Pass the cumulative results back into the next `suggest`.

**Do not pick or re-rank genes yourself.** A free tool-using agent was tested against this pipeline and
lost: hit@R5 0.230 vs 0.260 on GPT, and 0.209 vs 0.256 in the older Claude diagnostic. So relay the batch
as returned. You may explain the biology of the genes in it.

## Setup (once)

Every command below runs from the Waddington repository root — the checkout that contains this skill file,
two levels up from it — in the `waddington-bio` conda env. Substitute that path for `$WADDINGTON_HOME`.

```bash
cd "$WADDINGTON_HOME"
conda env create -f environment.yml                 # if `conda env list` has no waddington-bio
(cd frontend && npm install)                         # Node 20-24; the LLM calls go through it
node frontend/bin/waddington.js setup                # OAuth for the LLM (Codex/ChatGPT); interactive, once
conda run -n waddington-bio python3 -m waddington_select.suggest --dataset IFNG --n 5 --json   # check
```

`setup` needs the human to complete a browser login. If the check fails with an auth error, ask them to run
it; do not try to configure API keys some other way. The default LLM is `openai-codex/gpt-5.6-luna`.

## 1. Which phenotype

Benchmark phenotypes need no registration:

- `IFNG`, `IL2`
- `Sanchez21`, `Sanchez21_down`
- `Carnevale22`, `Scharenberg22`
- `Steinhart`
- `Replogle_K562_essential`, `Replogle_K562_gwps`

Use one of these only when the user is working on that exact screen. Otherwise, register their screen;
never map new biology onto a lookalike benchmark. List everything rankable with:

```bash
conda run -n waddington-bio python3 -m waddington_select.phenotype datasets --json
```

## 2. Register the scientist's screen (once per campaign)

Ask the scientist for each of these. Do not guess the hit rule or the perturbation.

| Input | Flag | Notes |
|---|---|---|
| Name | `--name` | Letters, digits and underscores only; not a benchmark name. |
| Phenotype and readout | `--task`, `--measurement` | One sentence each. The LLM reads these. |
| Library file | `--genes-file` | MAGeCK `gene_summary`, or a CSV/TSV with a gene column. Every gene in it becomes a candidate. |
| Perturbation | `--modality KO\|CRISPRi\|CRISPRa\|unknown` | Sets the DepMap features (see the first note below). |
| Cells | `--cell-line` | E.g. `K562`, `"primary T cells"`. |
| Curated essential panel | `--curated-essential` | Only if the library is a curated set of essential genes. |
| Hit rule | `--hit-rule ...` | Fixed for the whole campaign; see section 4. |
| Tier | `--tier non_essential\|full_pool` | Default `non_essential`; see the second note below. |
| Batch size | `--batch` | Genes tested per round (default 32). |

```bash
conda run -n waddington-bio python3 -m waddington_select.phenotype register --name MyScreen \
    --genes-file /path/to/library.csv --task "genes whose knockout reduces IL-2 secretion" \
    --measurement "IL-2 FACS sorting, sgRNA log-fold-change" --modality KO --cell-line "primary T cells" \
    --hit-rule fdr --max-fdr 0.05 --direction down --batch 64 --json
```

The JSON output reports these fields:

- `n_genes`: the candidates after the tier filter.
- `n_genes_with_known_features`: candidates with known features. Genes without features score from zeros,
  so tell the scientist if this is much lower than `n_genes`.
- `metadata`: the `modality`, `cell_line` and `curated_ess` as recorded.
- `depmap_policy`: `none`, `pan` or `pan_k562`.
- `hit_rule` and `hit_rule_text`
- `n_core_essential_removed`

Registration makes no network calls and is quick even on a genome-wide library; the frontend allows it 15
minutes. Confirm the reported `hit_rule_text`, `tier` and `metadata` back to the scientist before round 1 —
they are fixed for the campaign.

How the inputs are used:

- **DepMap features.** CRISPRa or a curated panel switches them off, because DepMap is knockout-derived and
  would restate or mislead the label. K562 keeps the pan-cancer features only. Everything else gets
  pan-cancer plus K562. `unknown` is treated as loss-of-function.
- **Tier.** `non_essential` (default) removes the 684 CEGv2 core-essential genes, so the budget goes to
  phenotype-specific hits; this is the paper's headline tier. `full_pool` keeps them.
- **Anchors.** `--anchors` is optional and not needed: the reported system ignores anchor features.
- **Name.** Re-registering the same name overwrites it. To change the hit rule or tier mid-campaign,
  register under a new name instead.

## 3. Each round: recommend a batch

`suggest` is **stateless**. Every call must pass all results so far:

- `--tested-hits`: every gene tested so far that was a hit.
- `--tested-misses`: every gene tested so far that was not a hit.
- `--exclude`: genes that must not come back, such as recommended genes the scientist decided not to test
  or genes with no usable measurement.

Genes passed in any of the three lists are never recommended again. Keep the campaign state in a file (see
section 5).

```bash
conda run -n waddington-bio python3 -m waddington_select.suggest --dataset MyScreen --json \
    [--n 64] [--tested-hits G1 G2 ...] [--tested-misses G3 ...] [--exclude G4 ...] [--trace]
```

The output is one JSON line: `{"dataset", "genes": [...], "info": {...}}`. The `info` fields are:

- `round`: 1 with no feedback, else 2.
- `n_feedback`, `n_hits`.
- `unknown_genes`: supplied symbols not in the pool. Report these; usually a symbol mismatch.
- `route`, `memory_entries` (normally 4).
- `trace`: per-gene attribution. Present only with `--trace`, which you may use to explain picks.

A call can take a few minutes on a genome-wide pool, since it retrains the model and calls the LLM. Allow at
least 5 minutes. Leave `--skills` and `--reason-from-structure` off; neither beats the default.

Without `--n`, the batch is the registered `--batch` (or the benchmark's batch size). Hand the scientist
the batch in the order returned.

## 4. After the wet lab: readout → hits

```bash
conda run -n waddington-bio python3 -m waddington_select.ingest --dataset MyScreen \
    --file /path/to/round1_readout.csv --genes <THE BATCH THAT WAS TESTED> --json
```

The output looks like this:

```json
{"tested": 64, "hits": [...], "misses": [...], "n_hits": 5, "unknown_genes": [...],
 "total_hits": null, "rule": "FDR < 0.05 (down)", "scope": "batch", "pool_coverage": 0.004,
 "overrides_ignored": []}
```

- **The registered hit rule always applies.** `overrides_ignored` lists any options you passed that were
  ignored because of it. The rules are:
  - `label`: the readout has a hit/label column.
  - `threshold`: `|score| >= X`; `--direction up|down` for one side.
  - `fdr`: `fdr < q`; MAGeCK `pos|fdr` / `neg|fdr` are recognised.
  - `top_fraction`: the top x% of the whole screen.
- **`unknown_genes`** are batch genes with no measurement in the file. They are neither hits nor misses.
  Never record them as misses: get the measurement, or move them to `--exclude`.
- **`scope`** is `batch` for one round's readout and `full_screen` when the file covers at least 90% of the
  pool. On a `batch`, `total_hits` is `null`: the screen-wide hit count is unknown, so report progress as
  hits per gene tested, never as a fraction of all hits. A `top_fraction` rule needs the whole screen and is
  refused on a batch.

## 5. Campaign state

`suggest` keeps nothing between calls, so you must. Keep one JSON file per campaign, for example
`<name>-campaign.json`, holding `dataset`, `round`, `hits`, `misses`, `excluded` and each round's batch.
Update it after every `ingest`, and build the next `suggest` call from it. If you lose it, rebuild it from
the readouts rather than starting the campaign over: dropping earlier rounds both loses the feedback and
lets already-tested genes be recommended again.

## 6. Limits — say so rather than working around them

- **Single-gene only.** Two-gene (combinatorial) screens are refused by `register` and `suggest`. On that
  tier the system does not beat BioDiscoveryAgent, so it is kept as a benchmark, not offered for use. If a
  user wants gene pairs, tell them this instead of ranking pairs some other way.
- **Changing the hit rule or tier mid-campaign** invalidates comparisons across rounds. Register a new
  phenotype instead.
- **No ground truth.** `suggest` never knows which genes are hits; only the scientist's readout does. The
  benchmark-oracle commands (`waddington_select.simulate`, `.tools reveal`) are demos and must not stand in
  for a real result.
- **What the numbers mean.** hit@R5 0.260 is the mean over nine public screens of the share of a five-round
  budget that were hits, against 0.231 for the same model without the LLM and 0.077 for random picking. A
  new screen's own rate depends on its hit rate and library; do not promise a number.
- **Expected gain.** The LLM contributes about +0.03 hit@R5 over the ML prior alone. It acts as a verifier
  of the ranker, not as a discoverer, and it shifts picks toward recurrent hits.

## Conversational alternative

The scientist can run the same pipeline themselves, with guided onboarding, in a chat UI:

```bash
cd "$WADDINGTON_HOME/frontend" && node bin/waddington.js     # then Terminal or Web UI
```

Point them there when they would rather drive it than have you relay it. Its onboarding asks the same
questions as section 2.

## Reference

- `AGENTS.md` in the repository root: the same commands, plus the benchmark loop.
- `README.md`: setup, the tier and hit-rule rules, and where results live.
- `skills/gene-selection-agent/SKILL.md`: the free tool-using agent. Benchmark-only, and it loses. Do not
  use it for real selection.
