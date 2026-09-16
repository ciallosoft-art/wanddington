// Local-only HTTP chat UI over the same conversation as the CLI.
import { createServer } from "node:http";
import { randomBytes, timingSafeEqual } from "node:crypto";
import { readFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { respond } from "./converse.mjs";
import { autoRun, autoRunUpload } from "./autorun.mjs";
import { pickDefaultChatModel } from "./setup.mjs";

const INDEX_HTML = resolve(dirname(fileURLToPath(import.meta.url)), "web", "index.html");
const MAX_BODY_BYTES = 16 * 1024 * 1024;
const identifier = /^[A-Za-z0-9_]+$/;
const plainObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const fail = (message, status = 400) => Object.assign(new Error(message), { status });

async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += Buffer.byteLength(chunk);
    if (size > MAX_BODY_BYTES) throw fail("Upload exceeds the 16 MB limit", 413);
    chunks.push(Buffer.from(chunk));
  }
  let body;
  try { body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}"); }
  catch { throw fail("Invalid JSON body"); }
  if (!plainObject(body)) throw fail("Expected a JSON object");
  return body;
}

function safeIdentifier(value, field, optional = true) {
  if (optional && value == null) return value;
  if (typeof value !== "string" || !identifier.test(value) || value.length > 100) throw fail(`Invalid ${field}`);
  return value;
}

function uploadedFile(file) {
  if (file == null) return undefined;
  if (!plainObject(file) || typeof file.content !== "string") throw fail("Upload file content is required");
  const name = basename(String(file.name || "upload.csv").replaceAll("\\", "/"));
  if (!/\.(csv|txt|tsv)$/i.test(name)) throw fail("Upload a CSV, TSV, or TXT file");
  return { name, content: file.content };
}

function checkedState(value) {
  if (value != null && !plainObject(value)) throw fail("Invalid conversation state");
  const state = value || {};
  const st = {
    dataset: safeIdentifier(state.dataset, "dataset") ?? null,
    hits: state.hits ?? [], misses: state.misses ?? [],
    campaign: state.campaign ?? null, onboarding: state.onboarding ?? null,
    feedbackByDataset: state.feedbackByDataset ?? {},
  };
  if (![st.hits, st.misses].every((xs) => Array.isArray(xs) && xs.every((g) => typeof g === "string"))) {
    throw fail("Invalid gene feedback");
  }
  if (!plainObject(st.feedbackByDataset)) throw fail("Invalid feedbackByDataset");
  for (const [name, feedback] of Object.entries(st.feedbackByDataset)) {
    safeIdentifier(name, "feedback dataset", false);
    if (!plainObject(feedback) || ![feedback.hits, feedback.misses].every(
      (xs) => Array.isArray(xs) && xs.every((g) => typeof g === "string"),
    )) throw fail("Invalid dataset feedback");
  }
  if (st.campaign != null) {
    if (!plainObject(st.campaign)) throw fail("Invalid campaign");
    safeIdentifier(st.campaign.dataset, "campaign dataset", false);
    if (typeof st.campaign.startedAt !== "string" ||
        !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(st.campaign.startedAt)) {
      throw fail("Invalid campaign start time");
    }
  }
  if (st.onboarding != null) {
    if (!plainObject(st.onboarding)) throw fail("Invalid onboarding state");
    safeIdentifier(st.onboarding.name, "phenotype name");
    // Browser state cannot authorize reads of arbitrary server-side paths.
    if (st.onboarding.genesPath) throw fail("Use a file upload in the Web UI, not a local file path");
    st.onboarding.genesPath = null;
    if (st.onboarding.genesContent != null && typeof st.onboarding.genesContent !== "string") {
      throw fail("Invalid uploaded gene pool");
    }
    if (st.onboarding.genesName != null) {
      st.onboarding.genesName = uploadedFile({ name: st.onboarding.genesName, content: "" }).name;
    }
  }
  return st;
}

function authorizedRequest(req, url, token) {
  if (!["127.0.0.1", "::1", "::ffff:127.0.0.1"].includes(req.socket.remoteAddress)) return false;
  const port = req.socket.localPort;
  if (![ `127.0.0.1:${port}`, `localhost:${port}` ].includes(req.headers.host)) return false;
  if (req.headers.origin && req.headers.origin !== `http://${req.headers.host}`) return false;
  if (req.headers["sec-fetch-site"] && !["same-origin", "none"].includes(req.headers["sec-fetch-site"])) return false;
  if (!url.pathname.startsWith("/api/")) return true;
  const candidate = req.headers["x-waddington-token"] ||
    (req.method === "GET" && url.pathname === "/api/run-stream" ? url.searchParams.get("token") : null);
  if (typeof candidate !== "string") return false;
  const actual = Buffer.from(candidate), expected = Buffer.from(token);
  return actual.length === expected.length && timingSafeEqual(actual, expected);
}

// Dependency injection keeps the HTTP boundary testable without LLM or Python calls.
export function createWebHandler({ modelSpec, token = randomBytes(32).toString("hex"), services = {} } = {}) {
  const api = { respond, autoRun, autoRunUpload, ...services };
  return async (req, res) => {
    res.setHeader("cache-control", "no-store");
    res.setHeader("referrer-policy", "no-referrer");
    res.setHeader("x-content-type-options", "nosniff");
    res.setHeader("x-frame-options", "DENY");
    res.setHeader("content-security-policy", "frame-ancestors 'none'; object-src 'none'; base-uri 'none'");
    try {
      const url = new URL(req.url, "http://localhost");
      if (!authorizedRequest(req, url, token)) throw fail("Forbidden", 403);
      if (req.method === "GET" && ["/", "/index.html"].includes(url.pathname)) {
        res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
        res.end(readFileSync(INDEX_HTML, "utf8").replace("__WADDINGTON_WEB_TOKEN__", token));
        return;
      }
      const stream = (run) => {
        res.writeHead(200, {
          "content-type": "text/event-stream", "cache-control": "no-store",
          connection: "keep-alive", "x-accel-buffering": "no",
        });
        let alive = true;
        res.on("close", () => { alive = false; });
        const send = (event) => { if (alive) res.write(`data: ${JSON.stringify(event)}\n\n`); };
        Promise.resolve().then(() => run(send))
          .then((sum) => { send({ type: "done", ...sum }); res.end(); })
          .catch((err) => { send({ type: "error", text: String(err.message || err) }); res.end(); });
      };
      if (req.method === "GET" && url.pathname === "/api/run-stream") {
        const dataset = safeIdentifier(url.searchParams.get("dataset") || "IFNG", "dataset", false);
        const rounds = Math.max(1, Math.min(8, parseInt(url.searchParams.get("rounds") || "5", 10) || 5));
        const batchSize = url.searchParams.get("batch") ? parseInt(url.searchParams.get("batch"), 10) : undefined;
        stream((onStep) => api.autoRun({ dataset, rounds, batchSize, modelSpec, onStep }));
        return;
      }
      if (req.method === "POST" && url.pathname === "/api/run-upload") {
        const body = await readBody(req);
        const name = safeIdentifier(body.name, "phenotype name", false);
        const file = uploadedFile({ name: body.readoutName || "readout.csv", content: body.readoutContent });
        stream((onStep) => api.autoRunUpload({
          name, readoutContent: file.content, readoutName: file.name,
          task: body.task, measurement: body.measurement,
          anchors: Array.isArray(body.anchors) ? body.anchors
            : (body.anchors ? String(body.anchors).split(/[,\s]+/).filter(Boolean) : undefined),
          rounds: Math.max(1, Math.min(8, parseInt(body.rounds || "5", 10) || 5)),
          batchSize: body.batchSize ? parseInt(body.batchSize, 10) : undefined,
          modelSpec, onStep,
        }));
        return;
      }
      if (req.method === "POST" && url.pathname === "/api/message") {
        const body = await readBody(req);
        const st = checkedState(body.state);
        const file = uploadedFile(body.file);
        const message = String(body.message || "");
        const expectsFile = (st.onboarding?.active && st.onboarding.stage === "pool") ||
          (st.campaign?.active && st.campaign.stage === "awaiting_results");
        if (!file && expectsFile && (/\.(csv|txt|tsv)$/i.test(message.trim()) || /^(\/|~|\.\/)/.test(message.trim()))) {
          throw fail("Use a file upload in the Web UI, not a local file path");
        }
        const result = await api.respond(message, st, modelSpec, file ? { file } : {});
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(result));
        return;
      }
      res.writeHead(404);
      res.end("not found");
    } catch (error) {
      res.writeHead(error.status || 500, { "content-type": "application/json" });
      res.end(JSON.stringify({ kind: "error", text: error.message }));
    }
  };
}

export async function launchWebServer({ port = 3000, modelSpec, services } = {}) {
  const spec = modelSpec || pickDefaultChatModel();
  const server = createServer(createWebHandler({ modelSpec: spec, services }));
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => { server.removeListener("error", reject); resolve(); });
  });
  console.log(`\nWaddington Web UI → http://localhost:${server.address().port}   (model: ${spec})`);
  console.log("Press Ctrl+C to stop.\n");
  return server;
}
