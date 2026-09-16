import test from "node:test";
import assert from "node:assert/strict";
import { request } from "node:http";
import { launchWebServer } from "../src/web-server.mjs";

test("the local Web boundary authenticates API requests and rejects unsafe state", async t => {
  const calls = [];
  const server = await launchWebServer({ port: 0, modelSpec: "offline/model", services: {
    respond: async (message, state, _spec, opts) => {
      calls.push({ message, state, opts });
      return { kind: "chat", text: "offline", state };
    },
    autoRun: async () => { calls.push("run"); return { rounds: 1 }; },
    autoRunUpload: async () => { calls.push("upload"); return { rounds: 1 }; },
  } });
  t.after(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }));
  assert.equal(server.address().address, "127.0.0.1");
  const port = server.address().port;
  async function send(path, { method = "GET", headers = {}, data } = {}) {
    return new Promise((resolve, reject) => {
      const req = request({ host: "127.0.0.1", port, path, method, headers }, res => {
        let text = "";
        res.setEncoding("utf8");
        res.on("data", chunk => { text += chunk; });
        res.on("end", () => resolve({ status: res.statusCode, headers: res.headers, text }));
      });
      req.on("error", reject);
      req.end(data == null ? undefined : JSON.stringify(data));
    });
  }
  const page = await send("/");
  assert.equal(page.status, 200);
  assert.equal(page.headers["cache-control"], "no-store");
  const token = page.text.match(/const WEB_TOKEN = "([a-f0-9]{64})"/)[1];
  const authorized = { "content-type": "application/json", "x-waddington-token": token };

  await t.test("token, Host, Origin and browser request context are checked before invoking services", async () => {
    for (const headers of [
      {}, { ...authorized, "x-waddington-token": "wrong" },
      { ...authorized, host: `attacker.example:${port}` },
      { ...authorized, origin: "https://attacker.example" },
      { ...authorized, "sec-fetch-site": "cross-site" },
    ]) {
      assert.equal((await send("/api/message", { method: "POST", headers, data: { message: "test" } })).status, 403);
    }
    assert.equal((await send("/", { headers: { origin: "https://attacker.example" } })).status, 403);
    assert.equal((await send("/api/run-stream?dataset=IFNG")).status, 403);
    assert.equal(calls.length, 0);
  });

  await t.test("normal chat retains per-phenotype feedback", async () => {
    const state = { dataset: "IL2", hits: [], misses: [], feedbackByDataset: { IFNG: { hits: ["STAT1"], misses: [] } } };
    const result = await send("/api/message", { method: "POST", headers: authorized, data: { message: "test", state } });
    assert.equal(result.status, 200);
    assert.deepEqual(JSON.parse(result.text).state.feedbackByDataset, state.feedbackByDataset);
    assert.equal(calls.length, 1);
  });

  await t.test("browser state cannot authorize directory traversal or server file reads", async () => {
    const count = calls.length;
    const examples = [
      { message: "stop", state: { campaign: { active: true, dataset: "../../../escape", startedAt: "2026-09-16T00:00:00.000Z" } } },
      { message: "yes", state: { onboarding: { active: true, name: "Screen", stage: "building", genesPath: "/tmp/private.csv" } } },
      { message: "/tmp/private.csv", state: { onboarding: { active: true, name: "Screen", stage: "pool" } } },
      { message: "/tmp/private.csv", state: { campaign: { active: true, dataset: "IFNG", startedAt: "2026-09-16T00:00:00.000Z", stage: "awaiting_results" } } },
    ];
    for (const data of examples) assert.equal((await send("/api/message", { method: "POST", headers: authorized, data })).status, 400);
    assert.equal(calls.length, count);
  });

  await t.test("uploaded file content is still accepted", async () => {
    const data = { message: "", state: { onboarding: { active: true, stage: "pool", name: "Screen" } }, file: { name: "screen.tsv", content: "gene\tlabel\nSTAT1\t1" } };
    assert.equal((await send("/api/message", { method: "POST", headers: authorized, data })).status, 200);
    assert.deepEqual(calls.at(-1).opts.file, data.file);
  });

  await t.test("both streaming entry points accept the page capability", async () => {
    const stream = await send(`/api/run-stream?dataset=IFNG&token=${token}`);
    assert.equal(stream.status, 200);
    assert.match(stream.text, /"type":"done"/);
    const upload = await send("/api/run-upload", { method: "POST", headers: authorized, data: { name: "Screen", readoutName: "screen.csv", readoutContent: "Gene,label\nSTAT1,1" } });
    assert.equal(upload.status, 200);
    assert.match(upload.text, /"type":"done"/);
    assert.deepEqual(calls.slice(-2), ["run", "upload"]);
  });
});
