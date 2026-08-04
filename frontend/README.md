# Waddington frontend — conversational entry

A tool-less conversational shell over the Waddington **C-arm gene-selection pipeline**.

```bash
cd frontend
npm install
node bin/waddington.js
```

On launch it authorizes a model provider (Claude / Codex / Gemini) into Waddington's **own** OAuth store
at `~/.waddington/agent/auth.json` — self-contained, no dependency on any other tool — then lets you pick
**Terminal (CLI)** or **Web UI**. You chat in natural language; report wet-lab results and it adapts the
next round. (Set `WADDINGTON_REUSE_FEYNMAN=1` to instead reuse an existing `~/.feynman` token store.)

## Design (why it looks like this)

Three layers, **no agent tools in the loop**:

1. **pi-ai (tool-less LLM)** — `src/llm/complete.mjs`. Uses `@earendil-works/pi-ai`'s `completeSimple`
   with `@earendil-works/pi-coding-agent`'s `AuthStorage`/`ModelRegistry` (multi-provider OAuth **with
   auto-refresh**). This is the real reusable substrate — *pi-ai*, not feynman-the-agent. The LLM only
   routes intent (`src/intent.mjs`) and narrates; it never selects genes.
2. **conversation** — `src/converse.mjs` (shared by CLI `src/chat.mjs` and Web `src/web-server.mjs`).
3. **brain** — `src/brain.mjs` shells out to the Python C-arm pipeline
   (`python -m waddington_select.suggest --json` / `.simulate`), the benchmark system
   (reported leakage-free hit@R5 = 0.251). Gene selection is deterministic and lives in Python.

We deliberately removed tools: the free tool-using agent lost to this pipeline (0.209 vs ~0.25).

## Commands

| command | purpose |
|---|---|
| `node bin/waddington.js` | interactive: auth → CLI/Web |
| `node bin/waddington.js setup` | authorize providers |
| `node bin/waddington.js --web [--port N]` | Web UI directly |
| `node bin/waddington.js complete` | one-shot completion (stdin/`--prompt`); used by the Python `pi` backend |

## Config

- `--model provider/model` (or `WADDINGTON_CHAT_MODEL`) — conversation model (default: an authorized Claude Haiku).
- `WADDINGTON_PY` — how to invoke Python (default `conda run -n waddington-bio python3`).
- `WADDINGTON_AUTH_PATH` — OAuth/API token store. Default: `~/.waddington/agent/auth.json` (Waddington's
  own store, created on first login). Set `WADDINGTON_REUSE_FEYNMAN=1` to reuse `~/.feynman/agent/auth.json`
  when feynman is installed.
