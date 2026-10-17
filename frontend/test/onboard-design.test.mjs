import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

import { parseDesign, parseModality, designText } from "../src/onboard.mjs";

test("perturbation replies", () => {
  const cases = [
    ["KO", "KO"], ["CRISPR knockout", "KO"], ["Cas9 KO screen", "KO"], ["敲除", "KO"],
    ["CRISPRi", "CRISPRi"], ["dCas9-KRAB", "CRISPRi"], ["knockdown", "CRISPRi"],
    ["CRISPRa", "CRISPRa"], ["CRISPR activation", "CRISPRa"], ["dCas9-VP64", "CRISPRa"], ["over-expression", "CRISPRa"],
    ["unknown", "unknown"], ["not sure", "unknown"], ["不知道", "unknown"],
  ];
  for (const [m, want] of cases) assert.equal(parseModality(m), want, m);
  // "T cell activation" is a phenotype, not CRISPRa; a bare "CRISPR" or "dCas9" names no modality
  for (const m of ["", "T cell activation", "CRISPR", "dCas9", "yes"]) assert.equal(parseModality(m), null, m);
});

test("design replies", () => {
  const cases = [
    ["CRISPRa, primary T cells", { modality: "CRISPRa", cellLine: "primary T cells", curatedEss: false }],
    ["KO in iPSC-derived neurons", { modality: "KO", cellLine: "iPSC-derived neurons", curatedEss: false }],
    ["CRISPRi in k562 cells", { modality: "CRISPRi", cellLine: "K562", curatedEss: false }],
    ["CRISPRi, K562, curated essential panel", { modality: "CRISPRi", cellLine: "K562", curatedEss: true }],
    ["KO, HEK293T, not a curated panel", { modality: "KO", cellLine: "HEK293T", curatedEss: false }],
    ["CRISPRi", { modality: "CRISPRi", cellLine: null, curatedEss: false }],
    ["unknown, Jurkat", { modality: "unknown", cellLine: "Jurkat", curatedEss: false }],
  ];
  for (const [m, want] of cases) assert.deepEqual(parseDesign(m), want, m);
  assert.match(designText({ modality: "CRISPRa", cellLine: "K562", curatedEss: false }), /CRISPRa in K562.*`yes`/);
  assert.match(designText(null), /KO.*CRISPRi.*CRISPRa/);
});

// onboard.mjs with brain.mjs / the LLM replaced, so the final stage does not register anything for real
async function onboarding(t) {
  const registrations = [];
  const deps = {
    "./llm/complete.mjs": { complete: async () => "{}" },
    "./brain.mjs": { registerPhenotype: async (args) => { registrations.push(args); return { name: args.name, n_genes: 1 }; } },
  };
  const module = new vm.SourceTextModule(readFileSync(new URL("../src/onboard.mjs", import.meta.url), "utf8"));
  await module.link((name) => {
    assert.ok(deps[name], `unexpected import ${name}`);
    return new vm.SyntheticModule(Object.keys(deps[name]), function () {
      for (const [k, v] of Object.entries(deps[name])) this.setExport(k, v);
    });
  });
  await module.evaluate();
  return { api: module.namespace, registrations };
}

async function toDesign(api, fields) {
  const state = {};
  await api.startOnboarding(state, { name: "S", task: "IL-2 secretion", measurement: "FACS", ...fields }, null);
  await api.handleOnboarding(state, "", { file: { content: "Gene,Score\nSTAT1,1\n", name: "pool.csv" } }, null);
  await api.handleOnboarding(state, "yes", {}, null);                 // tier
  const r = await api.handleOnboarding(state, "label", {}, null);    // hit rule
  return { state, r };
}

test("the design stage asks, then registers the design without anchors", async t => {
  const { api, registrations } = await onboarding(t);
  const { state, r } = await toDesign(api, {});
  assert.equal(r.stage, "design");
  assert.equal(r.guess, null);
  let s = await api.handleOnboarding(state, "yes", {}, null);       // nothing to confirm yet
  assert.equal(s.stage, "design");
  s = await api.handleOnboarding(state, "CRISPRa, primary T cells", {}, null);
  assert.equal(s.stage, "done");
  const reg = registrations[0];
  assert.deepEqual([reg.modality, reg.cellLine, reg.curatedEss, reg.anchors], ["CRISPRa", "primary T cells", false, undefined]);
  assert.equal(reg.tier, "non_essential");
  assert.deepEqual(reg.hitRule, { type: "label" });
});

test("a design named in the first message is offered for confirmation", async t => {
  const { api, registrations } = await onboarding(t);
  const { state, r } = await toDesign(api, { text: "I ran a CRISPR knockout screen for IL-2 secretion in K562" });
  assert.deepEqual(r.guess, { modality: "KO", cellLine: "K562", curatedEss: false });
  assert.match(r.text, /KO in K562/);
  await api.handleOnboarding(state, "yes", {}, null);
  assert.deepEqual([registrations[0].modality, registrations[0].cellLine], ["KO", "K562"]);

  const again = await toDesign(api, { text: "a CRISPRi screen of T cell activation" });
  assert.equal(again.r.guess.modality, "CRISPRi");                   // "activation" here is the phenotype
  await api.handleOnboarding(again.state, "yes, in primary T cells", {}, null);
  assert.deepEqual([registrations[1].modality, registrations[1].cellLine], ["CRISPRi", "primary T cells"]);
});
