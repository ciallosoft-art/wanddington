// intent.mjs — turn a scientist's natural-language message into a structured action.
//
// This is the ONLY place the conversation LLM is used, and it is used tool-lessly: it reads the
// message + session state and returns a small JSON object. It never selects genes — that is the
// Python C-arm's job (brain.mjs). Keeping the LLM to intent-parsing + narration is exactly the
// "remove the tools" finding (the free tool-using agent lost to the pipeline, 0.209 vs 0.256).

import { complete } from "./llm/complete.mjs";
import { DATASETS } from "./brain.mjs";

export const DEFAULT_CHAT_MODEL = process.env.WADDINGTON_CHAT_MODEL || "anthropic/claude-haiku-4-5";

const SYSTEM = `You are the intake router for a CRISPR gene-selection assistant.
The gene selection itself is done by a separate deterministic model — your ONLY job is to read the
scientist's message plus the running session state and emit a single JSON object describing the
action. Do not list or invent genes yourself.

Supported phenotypes (dataset ids): ${DATASETS.join(", ")}.

Respond with ONE JSON object, nothing else:
{
  "action": "suggest" | "experiment" | "simulate" | "chat",
  "dataset": <one of the ids above, or null if not yet known>,
  "new_hits":   [<gene symbols the scientist reports were HITS in this message>],
  "new_misses": [<gene symbols the scientist reports were NON-hits in this message>],
  "n": <batch size / how many genes per round, or null for default>,
  "rounds": <number of rounds for an experiment/simulate, or null>,
  "reply": <a short natural-language reply to show the scientist; REQUIRED for action="chat",
            optional otherwise>
}

Guidance:
- "recommend / which genes / suggest (one batch)" → action="suggest".
- "run/start a (simulated) experiment / let's do a screen / I want to actually run it round by round"
  → action="experiment" (an INTERACTIVE campaign: propose a batch, the scientist commits, the
  phenotype is revealed, repeat). This is the default when they want to *do* an experiment.
- "show me how it would go / auto-demo / just simulate it end-to-end" → action="simulate" (non-interactive).
- If the phenotype is ambiguous or unsupported, use action="chat" and ask which supported phenotype
  they mean (list a few). Never guess a dataset that isn't in the list.
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
  const raw = await complete({ spec: modelSpec, systemPrompt: SYSTEM, prompt, temperature: 0, maxTokens: 500 });
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
    reply: parsed.reply || null,
  };
}
