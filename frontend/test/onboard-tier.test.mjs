import test from "node:test";
import assert from "node:assert/strict";

import { parseTier, parseHitRule, startOnboarding, handleOnboarding } from "../src/onboard.mjs";

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

  r = await handleOnboarding(state, "keep", {}, null);
  assert.equal(state.onboarding.tier, "full_pool");
  assert.equal(r.stage, "hits");               // then the hit rule, before any result is seen

  r = await handleOnboarding(state, "top 5%", {}, null);
  assert.equal(r.stage, "hits");               // a top-x% call needs the whole screen: refused
  assert.match(r.text, /whole screen/);
  assert.equal(state.onboarding.hitRule, null);

  r = await handleOnboarding(state, "fdr < 0.05 up", {}, null);
  assert.deepEqual(state.onboarding.hitRule, { type: "fdr", maxFdr: 0.05, direction: "up" });
  assert.equal(r.stage, "design");             // then the screen's design (onboard-design.test.mjs)
});

test("hit rule replies", () => {
  const cases = [
    ["label", { type: "label" }],
    ["my results have a hit column", { type: "label" }],
    ["|score| >= 1.5", { type: "threshold", threshold: 1.5, direction: "both" }],
    ["|lfc| ≥ 1", { type: "threshold", threshold: 1, direction: "both" }],
    ["score >= 2", { type: "threshold", threshold: 2, direction: "up" }],
    ["score <= -1.5", { type: "threshold", threshold: 1.5, direction: "down" }],
    ["fdr < 0.05", { type: "fdr", maxFdr: 0.05, direction: "both" }],
    ["FDR<0.1 下调", { type: "fdr", maxFdr: 0.1, direction: "down" }],
    ["top 5%", { type: "top_fraction", fraction: 0.05, direction: "both" }],
  ];
  for (const [m, want] of cases) assert.deepEqual(parseHitRule(m), want, m);
  for (const m of ["", "whatever", "fdr < 2", "|score| >= 0", "top 0%"]) assert.equal(parseHitRule(m), null, m);
});

test("a bare yes accepts the default tier", async () => {
  const state = {};
  await startOnboarding(state, { name: "S", task: "t", measurement: "m" }, null);
  await handleOnboarding(state, "", { file: { content: "Gene,Score\nSTAT1,1\n", name: "pool.csv" } }, null);
  await handleOnboarding(state, "yes", {}, null);
  assert.equal(state.onboarding.tier, "non_essential");
});
