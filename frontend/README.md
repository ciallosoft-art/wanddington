# Waddington frontend — conversational entry

A tool-less conversational shell over the Waddington **C-arm gene-selection pipeline**.

```bash
cd frontend
npm install
node bin/waddington.js
```

On launch it authorizes Codex (or an optional compatibility provider) into Waddington's **own** OAuth store
at `~/.waddington/agent/auth.json` — self-contained, no dependency on any other tool — then lets you pick
**Terminal (CLI)** or **Web UI**. You chat in natural language; report wet-lab results and it adapts the
next round. (Set `WADDINGTON_REUSE_FEYNMAN=1` to instead reuse an existing `~/.feynman` token store.)

The Web UI listens on `127.0.0.1` only. Its page supplies a per-launch access token to local API
requests, and the server rejects cross-origin requests. Reload an open page after restarting the
server. Use file uploads in the Web UI; server-side file paths remain available in the CLI.

Onboarding your own screen asks for a name, the phenotype, the library file (every gene in it becomes a
candidate), the single-gene tier — remove the 684 CEGv2 core-essential genes (default, the paper's main
tier) or keep the full pool — the hit rule, and the screen's design. The hit rule (`label`, `|score| >= 1.5`,
`fdr < 0.05`, optionally `up`/`down`) is fixed for the campaign; a `top 5%` rule is refused there, since
you upload one batch per round and it needs the whole screen. Progress is then shown as hits per tested
gene, because the screen-wide hit count is unknown. The design is the perturbation and the cells (e.g.
`CRISPRa, primary T cells`; a perturbation named in your first message is offered for confirmation), plus
`curated essential panel` if that is what the library is. It sets which DepMap essentiality features the
model may use: none for CRISPRa or a curated panel, pan-cancer only in K562, otherwise pan-cancer + K562;
`unknown` is treated as loss-of-function. Anchor genes are no longer asked for: the reported system drops
the anchor-relative features. The Web UI's "upload & run" replays a *completed* screen: it asks the
perturbation, the cells and the tier in dialogs and registers the top 5% of the whole screen as the hits
(CLI: `waddington run --upload FILE --name X --modality CRISPRa --cell-line "…"`).

Feedback is saved separately for each phenotype. Uploads preserve CSV/TSV/MAGeCK format; missing
genes or measurements must be supplied before a campaign advances. If the next recommendation
fails, the confirmed results stay saved and `commit` retries selection without counting them again.

## Design (why it looks like this)

Three layers, **no agent tools in the loop**:

1. **pi-ai (tool-less LLM)** — `src/llm/complete.mjs`. Uses `@earendil-works/pi-ai`'s `completeSimple`
   with `@earendil-works/pi-coding-agent`'s `AuthStorage`/`ModelRegistry` (multi-provider OAuth **with
   auto-refresh**). This is the real reusable substrate — *pi-ai*, not feynman-the-agent. The LLM only
   routes intent (`src/intent.mjs`) and narrates; it never selects genes.
2. **conversation** — `src/converse.mjs` (shared by CLI `src/chat.mjs` and Web `src/web-server.mjs`).
3. **brain** — `src/brain.mjs` shells out to the Python C-arm pipeline
   (`python -m waddington_select.suggest --json` / `.simulate`). The paper reports a historical
   Claude-backend hit@R5 of 0.253; active GPT results are tracked separately. Selection policy and
   state transitions live in Python, while LLM-backed recommendations can vary across API calls.

We deliberately removed tools: in the matched legacy 5-seed diagnostic, the free tool-using agent lost
to the constrained pipeline (0.209 vs 0.256).

## Commands

| command | purpose |
|---|---|
| `node bin/waddington.js` | interactive: auth → CLI/Web |
| `node bin/waddington.js setup` | authorize providers |
| `node bin/waddington.js --web [--port N]` | Web UI directly |
| `node bin/waddington.js complete` | one-shot completion (stdin/`--prompt`); used by the Python `pi` backend |

## Config

- `--model provider/model` (or `WADDINGTON_CHAT_MODEL`) — conversation model (default: `openai-codex/gpt-5.6-luna`).
- `--reasoning LEVEL` (or `WADDINGTON_PI_REASONING`) — Codex effort; default `none`. The flag applies
  to one-shot, interactive CLI/Web narration, and child Python calls.
- `WADDINGTON_PI_MODEL` — Python C-arm model; default `openai-codex/gpt-5.6-luna`.
- `WADDINGTON_PY` — how to invoke Python (default `conda run -n waddington-bio python3`).
- `WADDINGTON_AUTH_PATH` — OAuth/API token store. Default: `~/.waddington/agent/auth.json` (Waddington's
  own store, created on first login). Set `WADDINGTON_REUSE_FEYNMAN=1` to reuse `~/.feynman/agent/auth.json`
  when feynman is installed.
