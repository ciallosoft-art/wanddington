import test from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { tmpdir } from "node:os";
import vm from "node:vm";

async function harness(t, unknown = []) {
  const root = fs.mkdtempSync(path.join(tmpdir(), "waddington-upload-test-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const known = [], registrations = [], ingestions = [], events = [];
  const deps = {
    "node:fs": fs, "node:path": path,
    "./brain.mjs": {
      REPO_ROOT: root, BENCHMARK_DATASETS: ["IFNG"], getDatasets: async () => known,
      registerPhenotype: async (args) => {
        registrations.push(args); known.push(args.name);
        return { n_genes: 2, n_genes_with_known_features: 2 };
      },
      suggestGenes: async () => ({ genes: ["STAT1", "JAK2"], info: {} }),
      ingestResults: async (args) => {
        ingestions.push(args);
        return { hits: ["STAT1"], misses: unknown.length ? [] : ["JAK2"], totalHits: 1, unknown };
      },
    },
    "./intent.mjs": { routeIntent: async () => ({ action: "experiment" }) },
    "./onboard.mjs": { proposeAnchors: async () => ["IRF1"] },
  };
  const source = fs.readFileSync(new URL("../src/autorun.mjs", import.meta.url), "utf8");
  const module = new vm.SourceTextModule(source);
  await module.link((name) => {
    assert.ok(deps[name], `unexpected import ${name}`);
    const values = deps[name];
    return new vm.SyntheticModule(Object.keys(values), function () {
      for (const [key, value] of Object.entries(values)) this.setExport(key, value);
    });
  });
  await module.evaluate();
  return { api: module.namespace, root, registrations, ingestions, events, onStep: e => events.push(e) };
}

for (const ext of ["csv", "tsv", "txt"]) {
  test(`upload preserves ${ext} parsing through registration and reveal`, async t => {
    const h = await harness(t);
    const content = ext === "csv" ? "Gene,Score\nSTAT1,2\nJAK2,0\n" : "id\tpos|lfc\nSTAT1\t2\nJAK2\t0\n";
    const result = await h.api.autoRunUpload({ name: "Screen", readoutContent: content, readoutName: `screen.${ext.toUpperCase()}`, rounds: 1 });
    const saved = path.join(h.root, "workspace/data/user_phenotypes/Screen", `readout.${ext}`);
    assert.equal(h.registrations[0].genesName, `readout.${ext}`);
    assert.equal(h.ingestions[0].path, saved);
    assert.equal(h.api.readoutFor("Screen"), saved);
    assert.equal(fs.readFileSync(saved, "utf8"), content);
    assert.equal(result.cumulative, 1);
  });
}

test("CLI upload derives the format from its file path", async t => {
  const h = await harness(t);
  const source = path.join(h.root, "gene_summary.txt");
  fs.writeFileSync(source, "id\tpos|lfc\nSTAT1\t2\nJAK2\t0\n");
  await h.api.autoRunUpload({ name: "Screen", readoutFile: source, rounds: 1 });
  assert.equal(h.registrations[0].genesName, "readout.txt");
  assert.equal(path.extname(h.ingestions[0].path), ".txt");
});

test("format changes select the latest readout, not a stale CSV", async t => {
  const h = await harness(t);
  for (const ext of ["csv", "tsv"]) {
    await h.api.autoRunUpload({ name: "Screen", readoutContent: "fixture", readoutName: `screen.${ext}`, rounds: 1 });
  }
  assert.equal(path.extname(h.api.readoutFor("Screen")), ".tsv");
});

test("unknown measurements stop an automatic campaign before outcomes are counted", async t => {
  const h = await harness(t, ["JAK2"]);
  await assert.rejects(h.api.autoRunUpload({
    name: "Screen", readoutContent: "Gene,label\nSTAT1,1\n", rounds: 2, onStep: h.onStep,
  }), /missing genes or measurements/);
  assert.equal(h.ingestions.length, 1);
  assert.equal(h.events.some(e => e.type === "revealed" || e.type === "summary"), false);
});
