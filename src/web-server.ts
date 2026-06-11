import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { resolve } from "node:path";
import { randomUUID } from "node:crypto";

import { WebSocket, WebSocketServer } from "ws";

import { buildPiArgs, buildPiEnv, type PiRuntimeOptions, resolvePiPaths, toNodeImportSpecifier } from "./pi/runtime.js";
import { patchPiRuntimeNodeModules } from "./pi/runtime-patches.js";
import { ensureSupportedNodeVersion } from "./system/node-version.js";
import { resolveAllExecutables } from "./system/executables.js";
import { openDb, dbListConvs, dbCreateConv, dbUpdateConv, dbDeleteConv, dbUpsertMsg,
         dbListExperiments, dbUpsertExperiment, dbDeleteExperiment, dbUpdateExperimentRFS,
         dbUpdateExperimentFilterVerdict,
         dbUpsertSummary, dbListSummaries,
         dbIndexFts, dbFtsSearch, dbUpsertEmbedding, dbLoadEmbeddings } from "./web/db.js";
import { extractExperiment, buildContextPrefix } from "./web/extract.js";
import { generateSummary, buildSummaryContext } from "./web/summarize.js";
import { embed, dotSim, vecToBlob, blobToVec, rrf, EMBED_DIMS } from "./web/embed.js";
import { computeRFS } from "./web/rfs.js";
import { loadProtocolSpec, listProtocolSlugs } from "./web/protocol.js";
import { updateLeaderboard, loadLeaderboard, rankLeaderboard } from "./web/leaderboard.js";
import { generateHypotheses, getHypothesesForGene, formatHypothesesContext } from "./web/hypothesize.js";
import { dbListHypotheses, dbUpdateHypothesisConfidence, dbUpdateExperimentFailure } from "./web/db.js";
import { classifyFailure } from "./web/failure.js";
import { crystalliseSkill, listSkills, findMatchingSkill, formatSkillContext } from "./web/skills.js";
import { refineSkill } from "./web/skill-refine.js";
import { seedFromProtocol, extractAndStoreKg, queryNeighbourhood, kgStats } from "./web/knowledge-graph.js";
import { applyNegativeFilter } from "./web/negative-filter.js";
import { getPhenotypeProfile } from "./web/phenotype-mapper.js";
import { rankGenes } from "./web/gene-ranker.js";
import { planExperiments } from "./web/experiment-planner.js";

function readFrontendTemplate(appRoot: string): string {
	const distPath = resolve(appRoot, "dist", "web", "index.html");
	const srcPath  = resolve(appRoot, "src",  "web", "index.html");
	const htmlPath = existsSync(distPath) ? distPath : srcPath;
	if (!existsSync(htmlPath)) throw new Error(`Frontend HTML not found at ${htmlPath}`);
	return readFileSync(htmlPath, "utf8");
}

// ── REST helpers ──────────────────────────────────────────────────────────────

async function readBody(req: IncomingMessage): Promise<unknown> {
	return new Promise((resolve, reject) => {
		let raw = "";
		req.on("data", (chunk: Buffer) => { raw += chunk.toString(); });
		req.on("end", () => { try { resolve(JSON.parse(raw || "{}")); } catch { resolve({}); } });
		req.on("error", reject);
	});
}

type JsonBody = Record<string, unknown>;

// Find the best-matching ProtocolSpec for a given model+dataset combination.
// Matches by slug/title containing the model name AND dataset name.
function findMatchingProtocol(model: string | null, dataset: string | null) {
	if (!model && !dataset) return null;
	for (const slug of listProtocolSlugs()) {
		const spec = loadProtocolSpec(slug);
		if (!spec) continue;
		const modelOk = !model ||
			slug.toLowerCase().includes(model.toLowerCase()) ||
			(spec.paper_title ?? "").toLowerCase().includes(model.toLowerCase());
		const datasetOk = !dataset ||
			spec.dataset.name.toLowerCase().includes(dataset.toLowerCase()) ||
			dataset.toLowerCase().includes(spec.dataset.name.toLowerCase());
		if (modelOk && datasetOk) return spec;
	}
	return null;
}

async function handleApi(req: IncomingMessage, res: ServerResponse): Promise<void> {
	const method = (req.method ?? "GET").toUpperCase();
	const path   = (req.url ?? "").replace(/\?.*$/, "");

	res.setHeader("Content-Type", "application/json");

	// GET /api/convs — list all conversations with messages
	if (path === "/api/convs" && method === "GET") {
		res.end(JSON.stringify(dbListConvs()));
		return;
	}

	// POST /api/convs — create conversation
	if (path === "/api/convs" && method === "POST") {
		const b = (await readBody(req)) as JsonBody;
		const id   = (b.id   as string | undefined) ?? randomUUID();
		const name = (b.name as string | undefined) ?? "New conversation";
		const sf   = (b.sessionFile as string | null | undefined) ?? null;
		res.writeHead(201);
		res.end(JSON.stringify(dbCreateConv(id, name, sf)));
		return;
	}

	const mConv = path.match(/^\/api\/convs\/([^/]+)$/);
	const mMsg  = path.match(/^\/api\/convs\/([^/]+)\/messages$/);

	// PATCH /api/convs/:id — update name and/or sessionFile
	if (mConv && method === "PATCH") {
		const b     = (await readBody(req)) as JsonBody;
		const patch: { name?: string; session_file?: string | null } = {};
		if (b.name !== undefined)        patch.name         = b.name as string;
		if ("sessionFile" in b)          patch.session_file = (b.sessionFile as string | null) ?? null;
		dbUpdateConv(mConv[1], patch);
		res.end('{"ok":true}');
		return;
	}

	// DELETE /api/convs/:id
	if (mConv && method === "DELETE") {
		dbDeleteConv(mConv[1]);
		res.end('{"ok":true}');
		return;
	}

	// POST /api/convs/:id/messages — upsert a message
	if (mMsg && method === "POST") {
		const b    = (await readBody(req)) as JsonBody;
		const id   = (b.id      as string | undefined) ?? randomUUID();
		const role = (b.role    as string | undefined) ?? "user";
		const text = (b.content as string | undefined) ?? "";
		const atts = JSON.stringify(b.atts ?? []);
		dbUpsertMsg(id, mMsg[1], role, text, atts);
		res.writeHead(201);
		res.end('{"ok":true}');
		return;
	}

	// POST /api/convs/:id/extract — extract experiment from conversation messages
	const mExtract = path.match(/^\/api\/convs\/([^/]+)\/extract$/);
	if (mExtract && method === "POST") {
		const conv = dbListConvs().find(c => c.id === mExtract[1]);
		if (!conv) { res.writeHead(404); res.end('{"error":"not found"}'); return; }
		const extracted = extractExperiment(conv.msgs);
		if (!extracted) { res.end('{"ok":true,"experiment":null}'); return; }
		const exp = dbUpsertExperiment({
			id:      randomUUID(),
			conv_id: conv.id,
			gene:    extracted.gene,
			model:   extracted.model,
			dataset: extracted.dataset,
			metrics: JSON.stringify(extracted.metrics),
			notes:   null,
			rfs_json: null,
			failure_json: null,
			filter_verdict_json: null,
		});
		res.writeHead(201);
		res.end(JSON.stringify({ ok: true, experiment: exp }));
		return;
	}

	// POST /api/convs/:id/process — extract experiment + generate summary in one shot
	const mProcess = path.match(/^\/api\/convs\/([^/]+)\/process$/);
	if (mProcess && method === "POST") {
		const conv = dbListConvs().find(c => c.id === mProcess[1]);
		if (!conv) { res.writeHead(404); res.end('{"error":"not found"}'); return; }

		// Extract experiment
		let experiment = null;
		const extracted = extractExperiment(conv.msgs);
		if (extracted) {
			experiment = dbUpsertExperiment({
				id:      randomUUID(),
				conv_id: conv.id,
				gene:    extracted.gene,
				model:   extracted.model,
				dataset: extracted.dataset,
				metrics: JSON.stringify(extracted.metrics),
				notes:   null,
				rfs_json: null,
				failure_json: null,
				filter_verdict_json: null,
			});
		}

		// Generate + store summary
		const sr = generateSummary(conv.msgs);
		const summaryRec = sr.msgCount >= 2
			? dbUpsertSummary(conv.id, sr.msgCount, sr.summary, sr.title, sr.topics)
			: null;

		// Auto-rename if still "New conversation"
		const isDefault = conv.name === "New conversation" || conv.name === "新对话";
		if (isDefault && sr.title) {
			dbUpdateConv(conv.id, { name: sr.title });
		}

		// Build FTS index body: title + summary + topics + message content
		const ftsBody = [
			sr.title ?? "",
			sr.summary,
			sr.topics.join(" "),
			...conv.msgs.map(m => m.content),
		].join(" ").replace(/\s+/g, " ").slice(0, 8000);
		dbIndexFts(conv.id, ftsBody);

		// Fire-and-forget: generate embedding (async, does not block response)
		embed(ftsBody.slice(0, 512)).then(vec => {
			dbUpsertEmbedding(conv.id, vecToBlob(vec), EMBED_DIMS);
		}).catch(() => { /* embedding is optional — ignore failures */ });

		// Fire-and-forget: compute RFS + update leaderboard + generate hypotheses
		if (experiment) {
			const ourMetrics = JSON.parse(experiment.metrics || "{}") as Record<string, number>;
			const spec = findMatchingProtocol(experiment.model, experiment.dataset);

			// Synchronous failure classification (pattern-based, no LLM)
			const failure = classifyFailure(conv.msgs);
			if (failure) {
				try { dbUpdateExperimentFailure(experiment.id, JSON.stringify(failure)); } catch { /* best-effort */ }
			}

			computeRFS(spec, ourMetrics, { convMsgs: conv.msgs }).then(rfsResult => {
				dbUpdateExperimentRFS(experiment.id, JSON.stringify(rfsResult));

				// G3: NegativeFilter — runs after RFS so all signals are available
				if (experiment.gene) {
					try {
						const filterResult = applyNegativeFilter(
							experiment.gene,
							ourMetrics,
							rfsResult,
							failure ?? undefined,
						);
						dbUpdateExperimentFilterVerdict(experiment.id, JSON.stringify(filterResult));
					} catch { /* filter is best-effort */ }
				}

				if (experiment.dataset && experiment.model) {
					try {
						updateLeaderboard(
							experiment.dataset,
							experiment.model,
							ourMetrics,
							rfsResult.overall >= 0 ? rfsResult.overall : undefined,
							conv.id,
						);
					} catch { /* leaderboard update is best-effort */ }
					// Crystallise SKILL after RFS so best_rfs is captured
					if (experiment.gene) {
						try {
							crystalliseSkill(
								experiment.gene,
								experiment.model,
								experiment.dataset,
								ourMetrics,
								rfsResult.overall >= 0 ? rfsResult.overall : undefined,
							);
						} catch { /* skill crystallisation is best-effort */ }

						// F3: if a failure was detected, fire-and-forget skill refinement
						if (failure) {
							const existingSkill = findMatchingSkill(
								experiment.gene,
								experiment.model ?? undefined,
								experiment.dataset ?? undefined,
							);
							if (existingSkill) {
								refineSkill(
									existingSkill.slug,
									failure,
									conv.msgs,
									spec ?? undefined,
								).catch(() => { /* refinement is best-effort */ });
							}
						}
					}
				}
			}).catch(() => { /* RFS is optional */ });

			// Fire-and-forget: hypothesis generation (after experiment extracted)
			if (experiment.gene) {
				generateHypotheses(
					experiment.gene,
					experiment.model,
					experiment.dataset,
					JSON.parse(experiment.metrics || "{}") as Record<string, number>,
					conv.msgs,
					experiment.id,
					conv.id,
				).catch(() => { /* hypotheses are optional */ });
			}
		}

		// Fire-and-forget: extract KG entities if this was a /discuss or /replicate session
		const hasDiscuss = conv.msgs.some(m => m.role === "user" && /^\/(?:discuss|replicate)\s/i.test(m.content));
		if (hasDiscuss) {
			extractAndStoreKg(conv.msgs).catch(() => { /* kg extraction is best-effort */ });
		}

		res.writeHead(201);
		res.end(JSON.stringify({ ok: true, experiment, summary: summaryRec, newTitle: isDefault ? sr.title : null }));
		return;
	}

	// GET /api/summaries — list recent summaries
	if (path === "/api/summaries" && method === "GET") {
		const qs  = new URL(req.url ?? "", "http://x").searchParams;
		const lim = parseInt(qs.get("limit") ?? "50", 10);
		res.end(JSON.stringify(dbListSummaries({ limit: lim })));
		return;
	}

	// GET /api/search?q=... — RRF(FTS5 + vector) search over conversations
	if (path === "/api/search" && method === "GET") {
		const q   = new URL(req.url ?? "", "http://x").searchParams.get("q") ?? "";
		const lim = 10;
		if (!q.trim()) { res.end("[]"); return; }

		// Layer 1: BM25 full-text search
		const ftsHits = dbFtsSearch(q, lim * 2);

		// Layer 3: vector cosine search (if any embeddings stored)
		const vecHits: Array<{ id: string; rank: number }> = [];
		const stored = dbLoadEmbeddings();
		if (stored.length > 0) {
			// Run embed async but we need the result synchronously for the response.
			// Best-effort: use a cached query embedding if possible; otherwise skip vec.
			// We pre-compute inline — acceptable latency for < 200 stored convs.
			const qEmbed = await embed(q).catch(() => null);
			if (qEmbed) {
				const sims = stored.map(row => ({
					id: row.conv_id,
					sim: dotSim(qEmbed, blobToVec(row.vector)),
				}));
				sims.sort((a, b) => b.sim - a.sim);
				sims.slice(0, lim * 2).forEach((s, i) => vecHits.push({ id: s.id, rank: i + 1 }));
			}
		}

		// RRF fusion
		const fused = rrf([
			ftsHits.map(h => ({ id: h.conv_id, rank: h.rank })),
			vecHits,
		]).slice(0, lim);

		// Enrich with conversation metadata + summary snippet
		const summaries = dbListSummaries({ limit: 200 });
		const sumMap = new Map(summaries.map(s => [s.conv_id, s]));
		const convs  = dbListConvs();
		const convMap = new Map(convs.map(c => [c.id, c]));

		const results = fused.map(({ id, score }) => {
			const conv = convMap.get(id);
			const sum  = sumMap.get(id);
			return {
				conv_id: id,
				name:    conv?.name ?? id,
				snippet: sum?.summary?.slice(0, 120) ?? "",
				title:   sum?.title ?? null,
				score,
			};
		}).filter(r => convMap.has(r.conv_id));

		res.end(JSON.stringify(results));
		return;
	}

	// GET /api/experiments — list experiments, optional ?gene=X&model=Y
	if (path === "/api/experiments" && method === "GET") {
		const qs   = new URL(req.url ?? "", "http://x").searchParams;
		const gene  = qs.get("gene")  ?? undefined;
		const model = qs.get("model") ?? undefined;
		const lim   = parseInt(qs.get("limit") ?? "50", 10);
		res.end(JSON.stringify(dbListExperiments({ gene, model, limit: lim })));
		return;
	}

	// DELETE /api/experiments/:id
	const mExp = path.match(/^\/api\/experiments\/([^/]+)$/);
	if (mExp && method === "DELETE") {
		dbDeleteExperiment(mExp[1]);
		res.end('{"ok":true}');
		return;
	}

	// POST /api/papers/audit — extract ProtocolSpec from a local PDF
	// Body: { pdf_path: string, slug?: string, skip_tables?: boolean }
	if (path === "/api/papers/audit" && method === "POST") {
		const body = (await readBody(req)) as JsonBody;
		const pdfPath = typeof body.pdf_path === "string" ? body.pdf_path : null;
		if (!pdfPath) {
			res.writeHead(400); res.end('{"error":"pdf_path is required"}'); return;
		}
		const { auditPaper, slugFromFilename } = await import("./web/paper-audit.js");
		const slug = typeof body.slug === "string" && body.slug
			? body.slug
			: slugFromFilename(pdfPath);
		const skipTables = body.skip_tables === true;
		try {
			const spec = await auditPaper(pdfPath, slug, { skipTableExtraction: skipTables });
			res.writeHead(201);
			res.end(JSON.stringify({ ok: true, slug, spec }));
		} catch (err) {
			res.writeHead(500);
			res.end(JSON.stringify({ error: (err as Error).message }));
		}
		return;
	}

	// GET /api/papers — list available ProtocolSpec slugs
	if (path === "/api/papers" && method === "GET") {
		const { listProtocolSlugs, loadProtocolSpec } = await import("./web/protocol.js");
		const slugs = listProtocolSlugs();
		const papers = slugs.map(s => {
			const spec = loadProtocolSpec(s);
			return { slug: s, title: spec?.paper_title, dataset: spec?.dataset?.name, annotated_by: spec?.annotated_by };
		});
		res.end(JSON.stringify(papers));
		return;
	}

	// GET /api/papers/:slug — get a specific ProtocolSpec
	const mPaper = path.match(/^\/api\/papers\/([^/]+)$/);
	if (mPaper && method === "GET") {
		const spec = loadProtocolSpec(mPaper[1]);
		if (!spec) { res.writeHead(404); res.end('{"error":"not found"}'); return; }
		res.end(JSON.stringify(spec));
		return;
	}

	// GET /api/leaderboard?dataset=norman2019  — ranked entries (all or filtered by dataset)
	if (path === "/api/leaderboard" && method === "GET") {
		const ds = new URL(req.url ?? "", "http://x").searchParams.get("dataset") ?? undefined;
		const ranked = rankLeaderboard(ds);
		res.end(JSON.stringify({ dataset: ds ?? null, entries: ranked, raw: loadLeaderboard() }));
		return;
	}

	// GET /api/hypotheses?gene=CEBPE&limit=10 — list hypotheses, optionally filtered
	if (path === "/api/hypotheses" && method === "GET") {
		const qs = new URL(req.url ?? "", "http://x").searchParams;
		const gene = qs.get("gene") ?? undefined;
		const limit = parseInt(qs.get("limit") ?? "20", 10);
		const rows = dbListHypotheses({ gene, limit });
		res.end(JSON.stringify(rows));
		return;
	}

	// PATCH /api/hypotheses/:id — update confidence label
	const mHyp = path.match(/^\/api\/hypotheses\/([^/]+)$/);
	if (mHyp && method === "PATCH") {
		const body = await readBody(req) as JsonBody;
		const confidence = body["confidence"] as string;
		if (!["speculative", "supported", "refuted"].includes(confidence)) {
			res.writeHead(400); res.end('{"error":"confidence must be speculative|supported|refuted"}'); return;
		}
		dbUpdateHypothesisConfidence(mHyp[1]!, confidence as "speculative" | "supported" | "refuted");
		res.end('{"ok":true}');
		return;
	}

	// GET /api/graph?node=CEBPE&depth=2 — neighbourhood subgraph
	if (path === "/api/graph" && method === "GET") {
		const qs    = new URL(req.url ?? "", "http://x").searchParams;
		const node  = qs.get("node") ?? "";
		const depth = Math.min(parseInt(qs.get("depth") ?? "2", 10), 3);
		if (!node) { res.writeHead(400); res.end('{"error":"node param required"}'); return; }
		res.end(JSON.stringify(queryNeighbourhood(node, depth)));
		return;
	}

	// GET /api/graph/stats — node and edge counts
	if (path === "/api/graph/stats" && method === "GET") {
		res.end(JSON.stringify(kgStats()));
		return;
	}

	// POST /api/negative-filter — classify a single experiment result (G3)
	if (path === "/api/negative-filter" && method === "POST") {
		const body = await readBody(req) as {
			gene?: string;
			metrics?: Record<string, number>;
			rfs?: import("./web/rfs.js").RFSResult;
			failure?: import("./web/failure.js").FailureRecord;
		};
		if (!body.gene || !body.metrics || !body.rfs) {
			res.writeHead(400);
			res.end(JSON.stringify({ error: "gene, metrics and rfs are required" }));
			return;
		}
		const result = applyNegativeFilter(body.gene, body.metrics, body.rfs, body.failure ?? null);
		res.end(JSON.stringify(result));
		return;
	}

	// GET /api/gene/:gene/phenotypes — G4 PhenotypeMapper
	const mPhenotype = path.match(/^\/api\/gene\/([^/]+)\/phenotypes$/);
	if (mPhenotype && method === "GET") {
		const gene = decodeURIComponent(mPhenotype[1]!).toUpperCase();
		try {
			const profile = await getPhenotypeProfile(gene);
			res.end(JSON.stringify(profile));
		} catch (err) {
			res.writeHead(500);
			res.end(JSON.stringify({ error: (err as Error).message }));
		}
		return;
	}

	// POST /api/experiment-plan — G2 ExperimentPlanner
	if (path === "/api/experiment-plan" && method === "POST") {
		const body = await readBody(req) as {
			dataset?: string;
			total_budget?: number;
			batch_size?: number;
			rounds?: number;
			explore_ratio?: number;
		};
		if (!body.total_budget || body.total_budget < 1) {
			res.writeHead(400);
			res.end(JSON.stringify({ error: "total_budget is required and must be >= 1" }));
			return;
		}
		try {
			const plan = await planExperiments({
				dataset:       body.dataset,
				total_budget:  body.total_budget,
				batch_size:    body.batch_size,
				rounds:        body.rounds,
				explore_ratio: body.explore_ratio,
			});
			res.end(JSON.stringify(plan));
		} catch (err) {
			res.writeHead(500);
			res.end(JSON.stringify({ error: (err as Error).message }));
		}
		return;
	}

	// POST /api/gene-select — G1 GeneRanker
	if (path === "/api/gene-select" && method === "POST") {
		const body = await readBody(req) as {
			dataset?: string;
			exclude_genes?: string[];
			n?: number;
		};
		try {
			const ranked = await rankGenes({
				dataset:       body.dataset,
				exclude_genes: body.exclude_genes,
				n:             body.n,
			});
			res.end(JSON.stringify(ranked));
		} catch (err) {
			res.writeHead(500);
			res.end(JSON.stringify({ error: (err as Error).message }));
		}
		return;
	}

	// GET /api/skills?status=active — list all skills
	if (path === "/api/skills" && method === "GET") {
		const qs = new URL(req.url ?? "", "http://x").searchParams;
		const status = qs.get("status") as "active" | "deprecated" | null;
		const skills = listSkills(status ? { status } : undefined);
		res.end(JSON.stringify(skills));
		return;
	}

	// GET /api/skills/match?gene=CEBPE&model=GEARS&dataset=norman2019
	if (path === "/api/skills/match" && method === "GET") {
		const qs = new URL(req.url ?? "", "http://x").searchParams;
		const gene = qs.get("gene") ?? "";
		const model = qs.get("model") ?? undefined;
		const dataset = qs.get("dataset") ?? undefined;
		if (!gene) { res.writeHead(400); res.end('{"error":"gene required"}'); return; }
		const skill = findMatchingSkill(gene, model, dataset);
		res.end(JSON.stringify({ skill, context: skill ? formatSkillContext(skill) : null }));
		return;
	}

	res.writeHead(404);
	res.end('{"error":"not found"}');
}

// ── Web server launcher ───────────────────────────────────────────────────────

export async function launchWebServer(options: PiRuntimeOptions, port = 3000): Promise<void> {
	ensureSupportedNodeVersion();
	patchPiRuntimeNodeModules(options.appRoot);

	const paths = resolvePiPaths(options.appRoot);

	if (!existsSync(paths.piCliPath)) throw new Error(`Pi CLI not found: ${paths.piCliPath}`);
	if (!existsSync(paths.piMainPath)) throw new Error(`Pi main module not found: ${paths.piMainPath}`);

	const useBuiltPolyfill = existsSync(paths.promisePolyfillPath);
	const useDevPolyfill = !useBuiltPolyfill && existsSync(paths.promisePolyfillSourcePath) && existsSync(paths.tsxLoaderPath);
	if (!useBuiltPolyfill && !useDevPolyfill) throw new Error(`Promise polyfill not found: ${paths.promisePolyfillPath}`);

	const useBuiltWrapper = existsSync(paths.piCliWrapperPath);
	const useDevWrapper = !useBuiltWrapper && existsSync(paths.piCliWrapperSourcePath) && existsSync(paths.tsxLoaderPath);
	if (!useBuiltWrapper && !useDevWrapper) throw new Error(`Waddington Pi CLI wrapper not found: ${paths.piCliWrapperPath}`);

	const wrapperPath = useBuiltWrapper ? paths.piCliWrapperPath : paths.piCliWrapperSourcePath;
	const importArgs = useDevPolyfill
		? ["--import", toNodeImportSpecifier(paths.tsxLoaderPath), "--import", toNodeImportSpecifier(paths.promisePolyfillSourcePath)]
		: ["--import", toNodeImportSpecifier(paths.promisePolyfillPath)];

	const executables = await resolveAllExecutables();

	// ── SQLite persistence ────────────────────────────────────────────────────
	// DB lives in ~/.waddington/web.db (same dir as sessions/ and agent/)
	const dbPath = resolve(options.feynmanAgentDir, "..", "web.db");
	openDb(dbPath);

	// Seed knowledge graph from existing ProtocolSpecs (idempotent upserts)
	for (const slug of listProtocolSlugs()) {
		const spec = loadProtocolSpec(slug);
		if (spec) try { seedFromProtocol(spec); } catch { /* best-effort */ }
	}

	// Force RPC mode so Pi outputs structured JSONL instead of TUI escape codes.
	const rpcOptions: PiRuntimeOptions = { ...options, mode: "rpc" };

	const pi = spawn(
		process.execPath,
		[...importArgs, wrapperPath, paths.piMainPath, ...buildPiArgs(rpcOptions, paths)],
		{
			cwd: options.workingDir,
			stdio: ["pipe", "pipe", "pipe"],
			env: buildPiEnv(rpcOptions, paths, executables),
		},
	);

	// ── HTTP server ───────────────────────────────────────────────────────────

	const htmlTemplate = readFrontendTemplate(options.appRoot);
	let html = "";

	const httpServer = createServer((req: IncomingMessage, res: ServerResponse) => {
		const url = req.url ?? "/";

		// REST API routes
		if (url.startsWith("/api/")) {
			handleApi(req, res).catch((err: Error) => {
				res.writeHead(500, { "Content-Type": "application/json" });
				res.end(JSON.stringify({ error: err.message }));
			});
			return;
		}

		// Static HTML
		if (url === "/" || url === "/index.html") {
			res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
			res.end(html);
		} else {
			res.writeHead(404);
			res.end("Not found");
		}
	});

	// ── WebSocket server ──────────────────────────────────────────────────────

	const wss = new WebSocketServer({ noServer: true });

	httpServer.on("upgrade", (req, socket, head) => {
		if (req.url === "/ws") {
			wss.handleUpgrade(req, socket, head, (ws) => {
				wss.emit("connection", ws, req);
			});
		} else {
			(socket as { destroy(): void }).destroy();
		}
	});

	const clients = new Set<WebSocket>();

	const broadcast = (line: string): void => {
		for (const client of clients) {
			if (client.readyState === WebSocket.OPEN) client.send(line);
		}
	};

	const rl = createInterface({ input: pi.stdout! });
	rl.on("line", (line) => {
		if (line.trim()) broadcast(line);
	});

	pi.stderr?.on("data", (chunk: Buffer) => {
		process.stderr.write(chunk);
		const text = chunk.toString().trim();
		if (text) {
			for (const line of text.split("\n")) {
				const l = line.trim();
				if (l && /error|failed|exception/i.test(l)) {
					broadcast(JSON.stringify({ type: "pi_error", text: l }));
				}
			}
		}
	});

	function injectContext(message: string): string {
		const match = message.match(/^\/(?:perturb|benchmark|design)\s+([A-Z][A-Z0-9]{1,11}(?:[+,][A-Z][A-Z0-9]{1,11})*)/i);
		const isDiscuss = /^\/(?:discuss|replicate)\s/i.test(message);
		if (!match && !isDiscuss) return message;

		let prefix = "";

		if (match) {
			const gene = match[1].toUpperCase();
			const exps = dbListExperiments({ gene, limit: 4 });
			if (exps.length) prefix += buildContextPrefix(gene, exps);

			// Inject SKILL context if a matching skill exists
			const skill = findMatchingSkill(gene);
			if (skill) prefix += formatSkillContext(skill) + "\n\n";

			// Inject relevant hypotheses (supported ones first)
			const hyps = getHypothesesForGene(gene, 3);
			if (hyps.length) prefix += formatHypothesesContext(hyps) + "\n\n";

			const sums = dbListSummaries({ limit: 50 });
			const sumCtx = buildSummaryContext(sums, [gene]);
			if (sumCtx) prefix += sumCtx;
		}

		if (isDiscuss) {
			const sums = dbListSummaries({ limit: 50 });
			const sumCtx = buildSummaryContext(sums, ["discuss", "replicate"]);
			if (sumCtx) prefix += sumCtx;
		}

		return prefix ? prefix + message : message;
	}

	wss.on("connection", (ws: WebSocket) => {
		clients.add(ws);

		ws.on("message", (data) => {
			const raw = Buffer.isBuffer(data) ? data.toString() : String(data);
			let forwarded = raw;
			try {
				const cmd = JSON.parse(raw) as Record<string, unknown>;
				if (cmd.type === "prompt" && typeof cmd.message === "string") {
					const enriched = injectContext(cmd.message);
					if (enriched !== cmd.message) {
						forwarded = JSON.stringify({ ...cmd, message: enriched });
					}
				}
			} catch { /* not JSON, forward as-is */ }
			if (pi.stdin && !pi.stdin.destroyed) {
				pi.stdin.write(forwarded + "\n");
			}
		});

		ws.on("close", () => { clients.delete(ws); });
		ws.on("error", () => { clients.delete(ws); });
	});

	// ── Start listening (auto-find free port) ─────────────────────────────────

	const boundPort = await new Promise<number>((resolve, reject) => {
		const tryListen = (p: number): void => {
			httpServer.once("error", (err: NodeJS.ErrnoException) => {
				if (err.code === "EADDRINUSE" && p < port + 10) {
					console.warn(`Port ${p} in use, trying ${p + 1}…`);
					tryListen(p + 1);
				} else {
					reject(err);
				}
			});
			httpServer.listen(p, "127.0.0.1", () => resolve(p));
		};
		tryListen(port);
	});

	html = htmlTemplate.replace("__PORT__", String(boundPort));

	const url = `http://localhost:${boundPort}`;
	console.log(`\nWaddington web UI: ${url}\n`);

	const { exec } = await import("node:child_process");
	const opener = process.platform === "darwin" ? "open" : process.platform === "win32" ? "start" : "xdg-open";
	exec(`${opener} ${url}`, () => { /* ignore errors */ });

	await new Promise<void>((resolvePromise, reject) => {
		pi.on("error", reject);
		pi.on("exit", (code) => {
			process.exitCode = code ?? 0;
			resolvePromise();
		});
	});
}
