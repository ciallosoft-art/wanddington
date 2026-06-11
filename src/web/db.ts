import Database from "better-sqlite3";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

// ── Types ────────────────────────────────────────────────────────────────────

export interface SummaryRecord {
	conv_id: string;
	msg_count: number;
	summary: string;
	title: string | null;
	topics_json: string;  // JSON string[]
	updated_at: number;
}

export interface ExpRecord {
	id: string;
	conv_id: string | null;
	gene: string | null;
	model: string | null;
	dataset: string | null;
	metrics: string;            // JSON
	notes: string | null;
	rfs_json: string | null;         // JSON-serialised RFSResult, computed async
	failure_json: string | null;     // JSON-serialised FailureRecord, computed async
	filter_verdict_json: string | null; // JSON-serialised FilterResult (G3 NegativeFilter)
	created_at: number;
}

export interface DbConv {
	id: string;
	session_file: string | null;
	name: string;
	created_at: number;
	updated_at: number;
}

interface RawMsg {
	id: string;
	conv_id: string;
	role: string;
	content: string;
	atts_json: string;
	created_at: number;
}

export interface ConvMsg {
	id: string;
	role: string;
	content: string;
	atts: unknown[];
}

export interface DbConvFull extends DbConv {
	msgs: ConvMsg[];
}

// ── Singleton ─────────────────────────────────────────────────────────────────

let _db: Database.Database;

export function getDb(): Database.Database { return _db; }

export function openDb(dbPath: string): void {
	mkdirSync(dirname(dbPath), { recursive: true });
	_db = new Database(dbPath);
	_db.pragma("journal_mode = WAL");
	_db.pragma("foreign_keys = ON");
	_db.exec(`
		CREATE TABLE IF NOT EXISTS conversations (
			id           TEXT    PRIMARY KEY,
			session_file TEXT,
			name         TEXT    NOT NULL DEFAULT 'New conversation',
			created_at   INTEGER NOT NULL,
			updated_at   INTEGER NOT NULL
		);
		CREATE TABLE IF NOT EXISTS messages (
			id         TEXT    PRIMARY KEY,
			conv_id    TEXT    NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
			role       TEXT    NOT NULL,
			content    TEXT    NOT NULL DEFAULT '',
			atts_json  TEXT    NOT NULL DEFAULT '[]',
			created_at INTEGER NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conv_id, created_at);
		CREATE INDEX IF NOT EXISTS idx_conv_upd ON conversations(updated_at DESC);
		CREATE TABLE IF NOT EXISTS experiments (
			id         TEXT    PRIMARY KEY,
			conv_id    TEXT    REFERENCES conversations(id) ON DELETE SET NULL,
			gene       TEXT,
			model      TEXT,
			dataset    TEXT,
			metrics    TEXT    NOT NULL DEFAULT '{}',
			notes      TEXT,
			created_at INTEGER NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_exp_gene ON experiments(gene);
		CREATE INDEX IF NOT EXISTS idx_exp_time ON experiments(created_at DESC);
		CREATE TABLE IF NOT EXISTS summaries (
			conv_id    TEXT    PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
			msg_count  INTEGER NOT NULL,
			summary    TEXT    NOT NULL,
			title      TEXT,
			topics_json TEXT   NOT NULL DEFAULT '[]',
			updated_at INTEGER NOT NULL
		);
		CREATE VIRTUAL TABLE IF NOT EXISTS fts_conv USING fts5(
			conv_id UNINDEXED,
			body,
			tokenize='unicode61 remove_diacritics 1'
		);
		CREATE TABLE IF NOT EXISTS embeddings (
			conv_id    TEXT    PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
			vector     BLOB    NOT NULL,
			dims       INTEGER NOT NULL,
			model      TEXT    NOT NULL DEFAULT 'all-MiniLM-L6-v2',
			updated_at INTEGER NOT NULL
		);
	`);
	// Migrations for existing databases
	try { _db.exec("ALTER TABLE experiments ADD COLUMN rfs_json TEXT"); } catch { /* already exists */ }
	try { _db.exec("ALTER TABLE experiments ADD COLUMN failure_json TEXT"); } catch { /* already exists */ }
	try { _db.exec("ALTER TABLE experiments ADD COLUMN filter_verdict_json TEXT"); } catch { /* already exists */ }

	// Knowledge graph tables (H1, idempotent)
	_db.exec(`
		CREATE TABLE IF NOT EXISTS kg_nodes (
			id         TEXT PRIMARY KEY,   -- e.g. "paper:gears-2024", "gene:CEBPE"
			type       TEXT NOT NULL,      -- Paper | Model | Gene | Dataset | Metric | CellType
			label      TEXT NOT NULL,      -- human-readable name
			props_json TEXT NOT NULL DEFAULT '{}'  -- extra attributes as JSON
		);
		CREATE INDEX IF NOT EXISTS idx_kg_node_type  ON kg_nodes(type);
		CREATE INDEX IF NOT EXISTS idx_kg_node_label ON kg_nodes(label);

		CREATE TABLE IF NOT EXISTS kg_edges (
			id       TEXT PRIMARY KEY,
			src      TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
			rel      TEXT NOT NULL,   -- benchmarks_on | claims | evaluated_by | perturbed_in | uses_metric
			dst      TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
			props_json TEXT NOT NULL DEFAULT '{}'
		);
		CREATE INDEX IF NOT EXISTS idx_kg_edge_src ON kg_edges(src);
		CREATE INDEX IF NOT EXISTS idx_kg_edge_dst ON kg_edges(dst);
		CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_edge_uniq ON kg_edges(src, rel, dst);
	`);

	// Hypotheses table (created in E2, idempotent)
	_db.exec(`
		CREATE TABLE IF NOT EXISTS hypotheses (
			id            TEXT    PRIMARY KEY,
			gene          TEXT    NOT NULL,
			related_genes TEXT    NOT NULL DEFAULT '[]',  -- JSON string[]
			model         TEXT,
			dataset       TEXT,
			hypothesis    TEXT    NOT NULL,
			evidence      TEXT    NOT NULL DEFAULT '',
			confidence    TEXT    NOT NULL DEFAULT 'speculative',  -- speculative | supported | refuted
			exp_id        TEXT    REFERENCES experiments(id) ON DELETE SET NULL,
			conv_id       TEXT    REFERENCES conversations(id) ON DELETE SET NULL,
			created_at    INTEGER NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_hyp_gene ON hypotheses(gene);
		CREATE INDEX IF NOT EXISTS idx_hyp_time ON hypotheses(created_at DESC);
	`);
}

// ── Conversations ─────────────────────────────────────────────────────────────

const parseMsg = (m: RawMsg): ConvMsg => ({
	id: m.id,
	role: m.role,
	content: m.content,
	atts: JSON.parse(m.atts_json) as unknown[],
});

export function dbListConvs(): DbConvFull[] {
	const convs = _db.prepare(
		"SELECT * FROM conversations ORDER BY updated_at DESC",
	).all() as DbConv[];

	const getMsgs = _db.prepare(
		"SELECT * FROM messages WHERE conv_id = ? ORDER BY created_at ASC",
	);

	return convs.map((c) => ({
		...c,
		msgs: (getMsgs.all(c.id) as RawMsg[]).map(parseMsg),
	}));
}

export function dbCreateConv(
	id: string,
	name: string,
	sessionFile: string | null = null,
): DbConv {
	const now = Date.now();
	_db.prepare(`
		INSERT OR IGNORE INTO conversations (id, session_file, name, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?)
	`).run(id, sessionFile, name, now, now);
	return { id, session_file: sessionFile, name, created_at: now, updated_at: now };
}

export function dbUpdateConv(
	id: string,
	patch: { name?: string; session_file?: string | null },
): void {
	const now = Date.now();
	const cols: string[] = ["updated_at = ?"];
	const vals: unknown[] = [now];
	if (patch.name !== undefined) { cols.push("name = ?"); vals.push(patch.name); }
	if ("session_file" in patch) { cols.push("session_file = ?"); vals.push(patch.session_file ?? null); }
	vals.push(id);
	_db.prepare(`UPDATE conversations SET ${cols.join(", ")} WHERE id = ?`).run(...vals);
}

export function dbDeleteConv(id: string): void {
	_db.prepare("DELETE FROM conversations WHERE id = ?").run(id);
}

// ── Messages ──────────────────────────────────────────────────────────────────

// ── Summaries ─────────────────────────────────────────────────────────────────

export function dbUpsertSummary(
	convId: string,
	msgCount: number,
	summary: string,
	title: string | null,
	topics: string[],
): SummaryRecord {
	const now = Date.now();
	const topicsJson = JSON.stringify(topics);
	_db.prepare(`
		INSERT INTO summaries (conv_id, msg_count, summary, title, topics_json, updated_at)
		VALUES (?, ?, ?, ?, ?, ?)
		ON CONFLICT(conv_id) DO UPDATE SET
			msg_count = excluded.msg_count, summary = excluded.summary,
			title = excluded.title, topics_json = excluded.topics_json, updated_at = excluded.updated_at
	`).run(convId, msgCount, summary, title ?? null, topicsJson, now);
	return { conv_id: convId, msg_count: msgCount, summary, title: title ?? null, topics_json: topicsJson, updated_at: now };
}

export function dbListSummaries(filter?: { limit?: number }): SummaryRecord[] {
	let sql = "SELECT * FROM summaries ORDER BY updated_at DESC";
	if (filter?.limit) sql += ` LIMIT ${filter.limit}`;
	return _db.prepare(sql).all() as SummaryRecord[];
}

export function dbGetSummary(convId: string): SummaryRecord | null {
	return (_db.prepare("SELECT * FROM summaries WHERE conv_id = ?").get(convId) as SummaryRecord | undefined) ?? null;
}

// ── Experiments ───────────────────────────────────────────────────────────────

export function dbListExperiments(filter?: { gene?: string; model?: string; limit?: number }): ExpRecord[] {
	let sql = "SELECT * FROM experiments";
	const params: unknown[] = [];
	const clauses: string[] = [];
	if (filter?.gene)  { clauses.push("gene = ?");  params.push(filter.gene); }
	if (filter?.model) { clauses.push("model = ?"); params.push(filter.model); }
	if (clauses.length) sql += " WHERE " + clauses.join(" AND ");
	sql += " ORDER BY created_at DESC";
	if (filter?.limit) { sql += " LIMIT ?"; params.push(filter.limit); }
	return _db.prepare(sql).all(...params) as ExpRecord[];
}

export function dbUpsertExperiment(exp: Omit<ExpRecord, "created_at">): ExpRecord {
	const now = Date.now();
	_db.prepare(`
		INSERT INTO experiments (id, conv_id, gene, model, dataset, metrics, notes, rfs_json, failure_json, filter_verdict_json, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			gene = excluded.gene, model = excluded.model, dataset = excluded.dataset,
			metrics = excluded.metrics, notes = excluded.notes,
			rfs_json = excluded.rfs_json, failure_json = excluded.failure_json,
			filter_verdict_json = excluded.filter_verdict_json
	`).run(exp.id, exp.conv_id ?? null, exp.gene ?? null, exp.model ?? null,
	       exp.dataset ?? null, exp.metrics, exp.notes ?? null,
	       exp.rfs_json ?? null, exp.failure_json ?? null, exp.filter_verdict_json ?? null, now);
	return { ...exp, created_at: now };
}

export function dbUpdateExperimentFailure(id: string, failureJson: string): void {
	_db.prepare("UPDATE experiments SET failure_json = ? WHERE id = ?").run(failureJson, id);
}

export function dbUpdateExperimentRFS(id: string, rfsJson: string): void {
	_db.prepare("UPDATE experiments SET rfs_json = ? WHERE id = ?").run(rfsJson, id);
}

export function dbUpdateExperimentFilterVerdict(id: string, verdictJson: string): void {
	_db.prepare("UPDATE experiments SET filter_verdict_json = ? WHERE id = ?").run(verdictJson, id);
}

export function dbDeleteExperiment(id: string): void {
	_db.prepare("DELETE FROM experiments WHERE id = ?").run(id);
}

// ── Messages ──────────────────────────────────────────────────────────────────

// ── FTS + Embeddings ──────────────────────────────────────────────────────────

/** Upsert the FTS index for a conversation. body = combined searchable text. */
export function dbIndexFts(convId: string, body: string): void {
	_db.prepare("DELETE FROM fts_conv WHERE conv_id = ?").run(convId);
	_db.prepare("INSERT INTO fts_conv (conv_id, body) VALUES (?, ?)").run(convId, body);
}

/** BM25-ranked FTS search. Returns conv_ids ordered by relevance. */
export function dbFtsSearch(query: string, limit = 20): Array<{ conv_id: string; rank: number }> {
	// Sanitize query for FTS5 (escape special chars)
	const safe = query.replace(/["*^()]/g, " ").trim();
	if (!safe) return [];
	try {
		const rows = _db.prepare(
			"SELECT conv_id, rank FROM fts_conv WHERE body MATCH ? ORDER BY rank LIMIT ?",
		).all(safe + "*", limit) as Array<{ conv_id: string; rank: number }>;
		// fts5 rank is negative (lower = better) — convert to 1-based rank
		return rows.map((r, i) => ({ conv_id: r.conv_id, rank: i + 1 }));
	} catch {
		return [];
	}
}

export interface EmbeddingRow { conv_id: string; vector: Buffer; dims: number; }

/** Store an embedding vector for a conversation. */
export function dbUpsertEmbedding(convId: string, vector: Buffer, dims: number): void {
	const now = Date.now();
	_db.prepare(`
		INSERT INTO embeddings (conv_id, vector, dims, updated_at)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(conv_id) DO UPDATE SET vector = excluded.vector, dims = excluded.dims, updated_at = excluded.updated_at
	`).run(convId, vector, dims, now);
}

/** Load all stored embeddings (for in-memory cosine search). */
export function dbLoadEmbeddings(): EmbeddingRow[] {
	return _db.prepare("SELECT conv_id, vector, dims FROM embeddings").all() as EmbeddingRow[];
}

// ── Hypotheses ────────────────────────────────────────────────────────────────

export interface HypothesisRecord {
	id: string;
	gene: string;
	related_genes: string;   // JSON string[]
	model: string | null;
	dataset: string | null;
	hypothesis: string;
	evidence: string;
	confidence: "speculative" | "supported" | "refuted";
	exp_id: string | null;
	conv_id: string | null;
	created_at: number;
}

export function dbInsertHypothesis(h: Omit<HypothesisRecord, "created_at">): HypothesisRecord {
	const now = Date.now();
	_db.prepare(`
		INSERT INTO hypotheses
			(id, gene, related_genes, model, dataset, hypothesis, evidence, confidence, exp_id, conv_id, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`).run(h.id, h.gene, h.related_genes, h.model ?? null, h.dataset ?? null,
	       h.hypothesis, h.evidence, h.confidence, h.exp_id ?? null, h.conv_id ?? null, now);
	return { ...h, created_at: now };
}

export function dbListHypotheses(filter?: { gene?: string; limit?: number }): HypothesisRecord[] {
	let sql = "SELECT * FROM hypotheses";
	const params: unknown[] = [];
	if (filter?.gene) { sql += " WHERE gene = ?"; params.push(filter.gene); }
	sql += " ORDER BY created_at DESC";
	if (filter?.limit) { sql += " LIMIT ?"; params.push(filter.limit); }
	return _db.prepare(sql).all(...params) as HypothesisRecord[];
}

export function dbUpdateHypothesisConfidence(id: string, confidence: "speculative" | "supported" | "refuted"): void {
	_db.prepare("UPDATE hypotheses SET confidence = ? WHERE id = ?").run(confidence, id);
}

// ── Messages ──────────────────────────────────────────────────────────────────

export function dbUpsertMsg(
	id: string,
	convId: string,
	role: string,
	content: string,
	attsJson: string,
): void {
	const now = Date.now();
	_db.prepare(`
		INSERT INTO messages (id, conv_id, role, content, atts_json, created_at)
		VALUES (?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET content = excluded.content, atts_json = excluded.atts_json
	`).run(id, convId, role, content, attsJson, now);
	_db.prepare("UPDATE conversations SET updated_at = ? WHERE id = ?").run(now, convId);
}
