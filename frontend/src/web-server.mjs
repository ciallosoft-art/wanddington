// web-server.mjs — minimal HTTP chat UI over the same tool-less conversation (converse.respond).
// Session state is held client-side and echoed back each request (no DB needed for v1).

import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { respond } from "./converse.mjs";
import { autoRun, autoRunUpload } from "./autorun.mjs";
import { pickDefaultChatModel } from "./setup.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const INDEX_HTML = resolve(HERE, "web", "index.html");

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (c) => (data += c));
    req.on("end", () => resolve(data));
    req.on("error", reject);
  });
}

export async function launchWebServer({ port = 3000, modelSpec } = {}) {
  const spec = modelSpec || pickDefaultChatModel();

  const server = createServer(async (req, res) => {
    try {
      if (req.method === "GET" && (req.url === "/" || req.url.startsWith("/index"))) {
        res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
        res.end(readFileSync(INDEX_HTML));
        return;
      }
      // Live step-by-step retrospective run (SSE): treat a completed screen's readout as ground truth,
      // auto-run the sequential campaign, and stream each step to the browser (Claude-Code-style).
      if (req.method === "GET" && req.url.startsWith("/api/run-stream")) {
        const u = new URL(req.url, "http://localhost");
        const dataset = u.searchParams.get("dataset") || "IFNG";
        const rounds = Math.max(1, Math.min(8, parseInt(u.searchParams.get("rounds") || "5", 10) || 5));
        const batchSize = u.searchParams.get("batch") ? parseInt(u.searchParams.get("batch"), 10) : undefined;
        res.writeHead(200, {
          "content-type": "text/event-stream", "cache-control": "no-cache",
          connection: "keep-alive", "x-accel-buffering": "no",
        });
        let alive = true;
        req.on("close", () => { alive = false; });
        const send = (e) => { if (alive) res.write(`data: ${JSON.stringify(e)}\n\n`); };
        autoRun({ dataset, rounds, batchSize, modelSpec: spec, onStep: send })
          .then((sum) => { send({ type: "done", ...sum }); res.end(); })
          .catch((err) => { send({ type: "error", text: String(err.message || err) }); res.end(); });
        return;
      }
      // Real production path (streamed over POST so the file body fits): register a brand-new screen
      // from the uploaded readout, then auto-run it — every step (upload → anchors → register → rounds)
      // streams back as SSE-style `data:` lines the browser reads from the fetch body.
      if (req.method === "POST" && req.url === "/api/run-upload") {
        const body = JSON.parse((await readBody(req)) || "{}");
        res.writeHead(200, {
          "content-type": "text/event-stream", "cache-control": "no-cache",
          connection: "keep-alive", "x-accel-buffering": "no",
        });
        let alive = true;
        req.on("close", () => { alive = false; });
        const send = (e) => { if (alive) res.write(`data: ${JSON.stringify(e)}\n\n`); };
        autoRunUpload({
          name: body.name, readoutContent: body.readoutContent, readoutName: body.readoutName,
          task: body.task, measurement: body.measurement,
          anchors: Array.isArray(body.anchors) ? body.anchors
            : (body.anchors ? String(body.anchors).split(/[,\s]+/).filter(Boolean) : undefined),
          rounds: Math.max(1, Math.min(8, parseInt(body.rounds || "5", 10) || 5)),
          batchSize: body.batchSize ? parseInt(body.batchSize, 10) : undefined,
          modelSpec: spec, onStep: send,
        }).then((sum) => { send({ type: "done", ...sum }); res.end(); })
          .catch((err) => { send({ type: "error", text: String(err.message || err) }); res.end(); });
        return;
      }
      if (req.method === "POST" && req.url === "/api/message") {
        const { message, state, file } = JSON.parse((await readBody(req)) || "{}");
        const st = {
          dataset: state?.dataset ?? null,
          hits: state?.hits ?? [],
          misses: state?.misses ?? [],
          campaign: state?.campaign ?? null,
          onboarding: state?.onboarding ?? null,
        };
        const result = await respond(String(message || ""), st, spec, file?.content != null ? { file } : {});
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(result));
        return;
      }
      res.writeHead(404);
      res.end("not found");
    } catch (e) {
      res.writeHead(500, { "content-type": "application/json" });
      res.end(JSON.stringify({ kind: "error", text: e.message }));
    }
  });

  await new Promise((r) => server.listen(port, r));
  console.log(`\nWaddington Web UI → http://localhost:${port}   (model: ${spec})`);
  console.log("Press Ctrl+C to stop.\n");
}
