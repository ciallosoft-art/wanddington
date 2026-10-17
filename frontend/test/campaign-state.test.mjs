// Run offline: node --experimental-vm-modules --test frontend/test/campaign-state.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const sourceDir = resolve(dirname(fileURLToPath(import.meta.url)), "../src");
const repoRoot = "/offline-waddington";

async function harness(overrides = {}) {
  const calls = { suggest: [], reveal: [], ingest: [], writes: [] };
  const brain = {
    REPO_ROOT: repoRoot,
    getDatasets: async () => ["IFNG", "IL2", "MyScreen"],
    simulateCampaign: async () => "offline simulation",
    suggestGenes: async (args) => {
      calls.suggest.push(structuredClone(args));
      if (overrides.suggest) return overrides.suggest(args, calls.suggest.length);
      return { genes: ["NEXT"], info: {} };
    },
    revealBatch: async (args) => {
      calls.reveal.push(structuredClone(args));
      return { hits: ["STAT1"], totalHits: 20 };
    },
    ingestResults: async (args) => {
      calls.ingest.push(structuredClone(args));
      if (overrides.ingest) return overrides.ingest(args, calls.ingest.length);
      return { hits: ["STAT1"], misses: ["ACTB"], unknown: [], totalHits: 20, method: "label:hit" };
    },
  };
  const replacements = {
    "./brain.mjs": brain,
    "./intent.mjs": {
      routeIntent: overrides.routeIntent || (async (dataset) => ({
        action: "suggest", dataset, new_hits: [], new_misses: [],
      })),
    },
    "./llm/complete.mjs": { complete: async () => "offline narration" },
    "./onboard.mjs": {
      startOnboarding: async () => ({ kind: "onboard", stage: "collect" }),
      handleOnboarding: overrides.handleOnboarding || (async (state) => {
        state.onboarding = null;
        state.dataset = "MyScreen";
        return { kind: "onboard", stage: "done", name: "MyScreen" };
      }),
    },
    "node:fs": {
      mkdirSync: () => {},
      writeFileSync: (path, content) => {
        if (overrides.failWrite) throw new Error("disk full");
        calls.writes.push({ path, content });
      },
    },
    "node:path": { join },
    "node:crypto": { randomUUID: () => "12345678-1234-4123-8123-123456789abc" },
  };
  const cache = new Map();
  async function load(name) {
    if (cache.has(name)) return cache.get(name);
    let mod;
    if (Object.hasOwn(replacements, name)) {
      const exports = replacements[name];
      mod = new vm.SyntheticModule(Object.keys(exports), function () {
        for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
      });
    } else {
      assert.ok(["./campaign.mjs", "./converse.mjs"].includes(name), `Unexpected import: ${name}`);
      mod = new vm.SourceTextModule(readFileSync(join(sourceDir, name), "utf8"));
    }
    cache.set(name, mod);
    await mod.link(load);
    return mod;
  }
  const mod = await load("./converse.mjs");
  await mod.evaluate();
  return { calls, respond: mod.namespace.respond, campaign: cache.get("./campaign.mjs").namespace };
}

function campaignState(mode = "oracle") {
  return {
    dataset: "IFNG", hits: [], misses: [],
    campaign: {
      active: true, mode, dataset: "IFNG", batchSize: 2,
      round: 1, maxRounds: 3, stage: mode === "upload" ? "awaiting_results" : "proposed",
      pending: ["STAT1", "ACTB"], testedHits: [], testedMisses: [],
      history: [], decisions: [], startedAt: "2026-09-16T00:00:00.000Z",
    },
  };
}

test("switching phenotypes isolates labels and restores them after a JSON session round trip", async () => {
  const h = await harness();
  let state = { dataset: "IFNG", hits: ["STAT1"], misses: ["ACTB"] };
  await h.respond("IL2", state, "offline/model");
  assert.deepEqual(h.calls.suggest[0].testedHits, []);
  assert.deepEqual(h.calls.suggest[0].testedMisses, []);
  state.hits.push("IL2RA");
  state = JSON.parse(JSON.stringify(state));
  await h.respond("IFNG", state, "offline/model");
  assert.deepEqual(h.calls.suggest[1].testedHits, ["STAT1"]);
  assert.deepEqual(h.calls.suggest[1].testedMisses, ["ACTB"]);
  assert.deepEqual(state.feedbackByDataset.IL2.hits, ["IL2RA"]);
});

test("registration switches to clean feedback without discarding the previous phenotype", async () => {
  const h = await harness();
  const state = { dataset: "IFNG", hits: ["STAT1"], misses: ["ACTB"], onboarding: { active: true } };
  await h.respond("yes", state, "offline/model");
  assert.equal(state.dataset, "MyScreen");
  assert.deepEqual(state.hits, []);
  assert.deepEqual(state.misses, []);
  await h.respond("IFNG", state, "offline/model");
  assert.deepEqual(h.calls.suggest[0].testedHits, ["STAT1"]);
});

test("a failed next recommendation preserves results and retries without repeating the oracle", async () => {
  const h = await harness({ suggest: async (_args, attempt) => {
    if (attempt <= 2) throw new Error("temporary provider failure");
    return { genes: ["JAK2", "IRF1"], info: {} };
  } });
  let state = campaignState();
  const first = await h.respond("commit", state, "offline/model");
  assert.equal(first.stage, "await");
  assert.match(first.text, /results are saved/);
  assert.equal(state.campaign.stage, "recommendation_pending");
  assert.equal(state.campaign.round, 1);
  assert.equal(state.campaign.history.length, 1);
  assert.deepEqual(state.campaign.testedHits, ["STAT1"]);
  state = JSON.parse(JSON.stringify(first.state));
  await h.respond("commit", state, "offline/model");
  assert.equal(state.campaign.history.length, 1);
  assert.equal(state.campaign.round, 1);
  const retried = await h.respond("commit", state, "offline/model");
  assert.equal(retried.stage, "revealed");
  assert.equal(retried.round, 1);
  assert.equal(retried.nextRound, 2);
  assert.equal(retried.cumulative, 1);
  assert.equal(h.calls.reveal.length, 1);
  assert.equal(state.campaign.history.length, 1);
  assert.deepEqual(state.campaign.pending, ["JAK2", "IRF1"]);
  assert.deepEqual(h.calls.suggest[2].testedMisses, ["ACTB"]);
});

test("reupload after recommendation failure retries selection without ingesting results twice", async () => {
  const h = await harness({ suggest: async (_args, attempt) => {
    if (attempt === 1) throw new Error("temporary provider failure");
    return { genes: ["JAK2", "IRF1"], info: {} };
  } });
  const state = campaignState("upload");
  const options = { file: { name: "screen.csv", content: "Gene,hit\nSTAT1,1\nACTB,0" } };
  const first = await h.respond("", state, "offline/model", options);
  assert.equal(first.stage, "await");
  const next = await h.respond("", state, "offline/model", options);
  assert.equal(next.nextRound, 2);
  assert.equal(next.cumulative, 1);
  assert.equal(h.calls.ingest.length, 1);
  assert.equal(state.campaign.history.length, 1);
});

test("a readout missing tested genes is rejected before any labels or round state change", async () => {
  const h = await harness({ ingest: async () => ({
    hits: ["STAT1"], misses: ["ACTB"], unknown: ["ACTB"], totalHits: 20,
  }) });
  const state = campaignState("upload");
  const before = structuredClone(state);
  await assert.rejects(h.campaign.submitResults(state, { path: "/offline/results.csv" }), /missing tested genes: ACTB/);
  assert.deepEqual(state, before);
  assert.equal(h.calls.suggest.length, 0);
});

test("stopping after a failed recommendation saves the confirmed outcome exactly once", async () => {
  const h = await harness({ suggest: async () => { throw new Error("offline failure"); } });
  const state = campaignState();
  await h.respond("commit", state, "offline/model");
  const stopped = await h.respond("stop", state, "offline/model");
  assert.equal(stopped.cumulative, 1);
  assert.equal(stopped.history.length, 1);
  assert.equal(state.campaign, null);
  assert.equal(JSON.parse(h.calls.writes[0].content).rounds_run, 1);
});

test("client-controlled dataset and timestamp cannot escape the campaign artifact directory", async () => {
  const h = await harness();
  const state = campaignState();
  state.campaign.dataset = "../../../../../../tmp/outside";
  state.campaign.startedAt = "../../attacker-chosen";
  const result = h.campaign.finish(state);
  const written = h.calls.writes[0];
  assert.equal(dirname(written.path), join(repoRoot, "workspace/results/campaigns"));
  assert.equal(result.tracePath, written.path);
  assert.equal(JSON.parse(written.content).dataset, "../../../../../../tmp/outside");
});

test("a failed trace write does not advertise a nonexistent saved artifact", async () => {
  const h = await harness({ failWrite: true });
  assert.equal(h.campaign.finish(campaignState()).tracePath, null);
});

test("an uploaded batch passes the phenotype to ingest and, with no screen-wide total, reports hits per tested gene", async () => {
  const h = await harness({ ingest: async () => ({
    hits: ["STAT1"], misses: ["ACTB"], unknown: [], totalHits: null, method: "|score| >= 1.5",
    rule: "|score| >= 1.5", scope: "batch", overridesIgnored: ["top_ratio"],
  }) });
  const state = campaignState("upload");
  const r = await h.respond("", state, "offline/model", { file: { name: "round1.csv", content: "Gene,Score\nSTAT1,2\nACTB,0" } });
  assert.equal(h.calls.ingest[0].dataset, "IFNG");
  assert.equal(r.stage, "revealed");
  assert.equal(r.total, null);
  assert.equal(r.ratio, null);          // never a made-up hit@R5
  assert.equal(r.testedSoFar, 2);
  assert.equal(r.hitRate, 0.5);
  assert.equal(r.hitRule, "|score| >= 1.5");
  assert.deepEqual(r.overridesIgnored, ["top_ratio"]);
});
