import test from "node:test";
import assert from "node:assert/strict";

import { parseTier, startOnboarding, handleOnboarding } from "../src/onboard.mjs";

test("tier replies", async (t) => {
  await t.test("keeping the full pool", () => {
    for (const m of ["keep", "Keep them all", "full pool please", "no", "保留必需基因", "用全池"]) {
      assert.equal(parseTier(m), "full_pool", m);
    }
  });
  await t.test("removing the core-essential genes", () => {
    for (const m of ["remove", "exclude them", "default", "non-essential", "排除必需基因", "默认"]) {
      assert.equal(parseTier(m), "non_essential", m);
    }
  });
  await t.test("anything else is undecided (a bare yes is the caller's default)", () => {
    for (const m of ["", "maybe", "yes", "ok", null]) assert.equal(parseTier(m), null, String(m));
  });
});

test("onboarding asks for the tier after the gene pool", async () => {
  const state = {};
  const fields = { name: "MyScreen", task: "IL-2 secretion", measurement: "FACS" };
  let r = await startOnboarding(state, fields, null);
  assert.equal(r.stage, "pool");

  r = await handleOnboarding(state, "", { file: { content: "Gene,Score\nSTAT1,1\n", name: "pool.csv" } }, null);
  assert.equal(r.stage, "tier");
  assert.match(r.text, /core-essential/);

  r = await handleOnboarding(state, "maybe", {}, null);
  assert.equal(r.stage, "tier");               // undecided: ask again
  assert.equal(state.onboarding.tier, null);

  state.onboarding.proposed = ["STAT1"];      // skip the LLM anchor proposal
  r = await handleOnboarding(state, "keep", {}, null);
  assert.equal(state.onboarding.tier, "full_pool");
  assert.equal(r.stage, "anchors");
});

test("a bare yes accepts the default tier", async () => {
  const state = {};
  await startOnboarding(state, { name: "S", task: "t", measurement: "m" }, null);
  await handleOnboarding(state, "", { file: { content: "Gene,Score\nSTAT1,1\n", name: "pool.csv" } }, null);
  state.onboarding.proposed = ["STAT1"];
  await handleOnboarding(state, "yes", {}, null);
  assert.equal(state.onboarding.tier, "non_essential");
});
