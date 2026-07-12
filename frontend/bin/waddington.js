#!/usr/bin/env node
// Waddington conversational entry — auth → CLI/Web → tool-less pi-ai shell over the C-arm pipeline.
import { resolve, dirname } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";

const MIN_NODE_MAJOR = 20;
const MAX_NODE_MAJOR = 24;
const major = Number(process.versions.node.split(".")[0]) || 0;
if (major < MIN_NODE_MAJOR || major > MAX_NODE_MAJOR) {
  console.error(`waddington requires Node.js ${MIN_NODE_MAJOR}.x–${MAX_NODE_MAJOR}.x (detected ${process.versions.node}).`);
  process.exit(1);
}

const here = dirname(fileURLToPath(import.meta.url));
const { main } = await import(pathToFileURL(resolve(here, "..", "src", "index.mjs")).href);
try {
  await main();
} catch (err) {
  console.error(`\n[waddington] ${err?.message || err}`);
  process.exit(1);
}
