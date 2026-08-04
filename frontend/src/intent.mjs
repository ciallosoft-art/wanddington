// intent.mjs — turn a scientist's natural-language message into a structured action.
//
// This is the ONLY place the conversation LLM is used, and it is used tool-lessly: it reads the
// message + session state and returns a small JSON object. It never selects genes — that is the
// Python C-arm's job (brain.mjs). Keeping the LLM to intent-parsing + narration is exactly the
// "remove the tools" finding (the free tool-using agent lost to the pipeline, 0.209 vs 0.256).

import { complete } from "./llm/complete.mjs";
import { getDatasets } from "./brain.mjs";

export const DEFAULT_CHAT_MODEL = process.env.WADDINGTON_CHAT_MODEL || "anthropic/claude-haiku-4-5";

const SYSTEM = (DATASETS) => `You are **Waddington**, a conversational assistant for CRISPR
gene-selection. The gene selection itself is done by a separate deterministic model (the "C-arm
pipeline") — your ONLY job is to read the scientist's message plus the running session state and emit a
single JSON object describing the action. Do not list or invent genes yourself.

IDENTITY: You are Waddington. In any "reply" text you write, speak as Waddington in the first person.
NEVER identify yourself as Claude, Claude Code, an AI language model, Anthropic, GPT, or any underlying
model/vendor. If the scientist greets you or asks who you are, introduce yourself as Waddington — a
front-end to a CRISPR gene-selection pipeline — and briefly offer what you can do (suggest a batch of
genes for a phenotype, run an interactive round-by-round experiment, simulate a screen end-to-end, or
register their own screen).

Supported phenotypes (dataset ids): ${DATASETS.join(", ")}.

Respond with ONE JSON object, nothing else:
{
  "action": "suggest" | "experiment" | "simulate" | "register" | "chat",
  "dataset": <one of the ids above, or null if not yet known>,
  "new_hits":   [<gene symbols the scientist reports were HITS in this message>],
  "new_misses": [<gene symbols the scientist reports were NON-hits in this message>],
  "n": <batch size / how many genes per round, or null for default>,
  "rounds": <number of rounds for an experiment/simulate, or null>,
  "mode": "oracle" | "upload",
  "name": <for action="register": a short id for THEIR screen (e.g. "Ferroptosis_K562"), else null>,
  "task": <for action="register": one sentence on the phenotype they're screening, else null>,
  "measurement": <for action="register": what their readout measures, else null>,
  "reply": <a short natural-language reply to show the scientist; REQUIRED for action="chat",
            optional otherwise>
}

Guidance:
- "recommend / which genes / suggest (one batch)" → action="suggest".
- "run/start a (simulated) experiment / let's do a screen / I want to actually run it round by round"
  → action="experiment" (an INTERACTIVE campaign: propose a batch, the scientist commits, the
  phenotype is revealed, repeat). This is the default when they want to *do* an experiment.
- "mode": for action="experiment", set "upload" when the scientist will supply their OWN screen
  results each round (phrases: "my own data / real experiment / I'll upload results / my screen
  file / MAGeCK"); otherwise "oracle" (default — the benchmark truth stands in for the wet lab).
- "show me how it would go / auto-demo / just simulate it end-to-end" → action="simulate" (non-interactive).
- action="register" when the scientist wants to work on a phenotype that is NOT in the list above —
  i.e. their OWN screen ("onboard/register my screen", "I ran a screen for X", or they simply name a
  phenotype that isn't supported). Their screen must be onboarded (features built) before it can be
  ranked. Never force their new biology onto a lookalike benchmark dataset; use "register" instead.
- Only leave "dataset" set to an id from the list above. If the phenotype is theirs/new, set
  "dataset": null and use action="register".
- If the message is too vague to act on at all, use action="chat" and ask what they need.
- Extract gene symbols exactly as written (uppercase). Only fill new_hits/new_misses when the
  scientist is reporting experimental results from a previous round.`;

/** Best-effort extraction of the first JSON object in a string. */
function parseJson(text) {
  let t = text.trim();
  if (t.startsWith("```")) t = t.replace(/^```[a-zA-Z]*\n?/, "").replace(/\n?```$/, "").trim();
  try {
    return JSON.parse(t);
  } catch {
    const m = t.match(/\{[\s\S]*\}/);
    if (m) {
      try {
        return JSON.parse(m[0]);
      } catch {
        /* fall through */
      }
    }
  }
  return null;
}

/**
 * @param {string} message      the scientist's latest message
 * @param {object} state        { dataset, hits: string[], misses: string[] }
 * @param {string} [modelSpec]  "provider/model"
 */
export async function routeIntent(message, state, modelSpec = DEFAULT_CHAT_MODEL) {
  const stateStr = JSON.stringify({
    dataset: state.dataset ?? null,
    known_hits: state.hits ?? [],
    known_misses: state.misses ?? [],
  });
  const prompt = `Session state so far: ${stateStr}\n\nScientist's message: """${message}"""\n\nEmit the JSON action.`;
  const datasets = await getDatasets(); // includes any phenotype the scientist registered
  const raw = await complete({ spec: modelSpec, systemPrompt: SYSTEM(datasets), prompt, temperature: 0, maxTokens: 500 });
  const parsed = parseJson(raw);
  if (!parsed || typeof parsed !== "object") {
    return { action: "chat", dataset: state.dataset ?? null, new_hits: [], new_misses: [], reply: raw };
  }
  return {
    action: parsed.action || "chat",
    dataset: parsed.dataset ?? state.dataset ?? null,
    new_hits: Array.isArray(parsed.new_hits) ? parsed.new_hits.map((g) => String(g).toUpperCase()) : [],
    new_misses: Array.isArray(parsed.new_misses) ? parsed.new_misses.map((g) => String(g).toUpperCase()) : [],
    n: Number.isInteger(parsed.n) ? parsed.n : null,
    rounds: Number.isInteger(parsed.rounds) ? parsed.rounds : null,
    mode: parsed.mode === "upload" ? "upload" : "oracle",
    name: parsed.name || null,
    task: parsed.task || null,
    measurement: parsed.measurement || null,
    reply: parsed.reply || null,
  };
}
