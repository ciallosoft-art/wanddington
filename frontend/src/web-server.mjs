// web-server.mjs — minimal HTTP chat UI over the same tool-less conversation (converse.respond).
// Session state is held client-side and echoed back each request (no DB needed for v1).

import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { respond } from "./converse.mjs";
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
      if (req.method === "POST" && req.url === "/api/message") {
        const { message, state } = JSON.parse((await readBody(req)) || "{}");
        const st = {
          dataset: state?.dataset ?? null,
          hits: state?.hits ?? [],
          misses: state?.misses ?? [],
        };
        const result = await respond(String(message || ""), st, spec);
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
