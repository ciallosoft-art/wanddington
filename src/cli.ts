import "dotenv/config";

import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";

import { SettingsManager } from "@earendil-works/pi-coding-agent";

import { syncBundledAssets } from "./bootstrap/sync.js";
import {
	ensureWaddingtonHome,
	getDefaultSessionDir,
	getWaddingtonAgentDir,
	getWaddingtonHome,
} from "./config/paths.js";
import { select, isCancel } from "@clack/prompts";

import { launchPiChat } from "./pi/launch.js";
import { launchWebServer } from "./web-server.js";
import { installPackageSources, updateConfiguredPackages } from "./pi/package-ops.js";
import { MAX_NATIVE_PACKAGE_NODE_MAJOR } from "./pi/package-presets.js";
import {
	CORE_PACKAGE_SOURCES,
	resolvePackageUpdateSources,
} from "./pi/package-presets.js";
import { normalizeFeynmanSettings, normalizeThinkingLevel, type ThinkingLevel } from "./pi/settings.js";
import { applyFeynmanPackageManagerEnv } from "./pi/runtime.js";
import { setSearchProvider, clearSearchConfig, printSearchStatus } from "./search/commands.js";
import type { PiWebSearchProvider } from "./pi/web-access.js";
import { runDoctor } from "./setup/doctor.js";
import { runSetup } from "./setup/setup.js";
import { ASH, RESET, SAGE, printAsciiHeader, printInfo, printSection } from "./ui/terminal.js";

function printHelp(): void {
	printAsciiHeader([
		"Single-cell gene perturbation research agent.",
		"Use `waddington setup` on first run.",
	]);

	printSection("Getting Started");
	printInfo("waddington");
	printInfo("waddington setup");
	printInfo("waddington doctor");
	printInfo("waddington search status");

	printSection("Paper Workflows");
	[
		["/discuss <url|path>",  "Load and discuss a paper; extract code, data, methods"],
		["/replicate <url>",     "Automatic paper → install → run → compare pipeline"],
	].forEach(([u, d]) => {
		const pad = " ".repeat(Math.max(1, 32 - u.length));
		console.log(`  ${SAGE}${u}${RESET}${pad}${ASH}${d}${RESET}`);
	});

	printSection("Perturbation Workflows");
	[
		["/perturb <gene>",            "Predict transcriptomic effect of a KO/OE"],
		["/analyze <path.h5ad>",       "Inspect a single-cell dataset"],
		["/benchmark <model1> [m2]",   "Compare models on a dataset"],
		["/design <question>",         "Design an experiment from a biological question"],
		["/model install|list|info",   "Manage perturbation models"],
	].forEach(([u, d]) => {
		const pad = " ".repeat(Math.max(1, 32 - u.length));
		console.log(`  ${SAGE}${u}${RESET}${pad}${ASH}${d}${RESET}`);
	});

	printSection("Gene Selection (G1+G2)");
	[
		["suggest-genes <DS> --budget N", "Plan multi-round gene selection experiment"],
	].forEach(([u, d]) => {
		const pad = " ".repeat(Math.max(1, 32 - u.length));
		console.log(`  ${SAGE}${u}${RESET}${pad}${ASH}${d}${RESET}`);
	});
}

function loadPackageVersion(appRoot: string): { version?: string } {
	try {
		return JSON.parse(readFileSync(resolve(appRoot, "package.json"), "utf8")) as { version?: string };
	} catch {
		return {};
	}
}

export async function main(): Promise<void> {
	const here    = dirname(fileURLToPath(import.meta.url));
	const appRoot = resolve(here, "..");   // waddington project root

	const version          = loadPackageVersion(appRoot).version;
	const home             = getWaddingtonHome();
	const agentDir         = getWaddingtonAgentDir(home);
	const settingsPath     = resolve(agentDir, "settings.json");
	const bundledSettings  = resolve(appRoot, ".waddington", "settings.json");
	const authPath         = resolve(agentDir, "auth.json");
	const sessionDir       = getDefaultSessionDir(home);

	ensureWaddingtonHome(home);
	syncBundledAssets(appRoot, agentDir);

	// suggest-genes has its own arg parser — intercept before parseArgs sees unknown flags
	const rawArgv = process.argv.slice(2).filter(a => a !== "--");
	if (rawArgv[0] === "suggest-genes" || rawArgv[0] === "/suggest-genes") {
		// DB must be open before G1/G2 can query experiments / hypotheses
		const { openDb } = await import("./web/db.js");
		const { resolve: res } = await import("node:path");
		openDb(res(agentDir, "..", "web.db"));

		const { handleSuggestGenes } = await import("./web/suggest-genes.js");
		const plan = await handleSuggestGenes(rawArgv.join(" "));
		console.log(plan);
		return;
	}

	const { values, positionals } = parseArgs({
		args: process.argv.slice(2),
		allowPositionals: true,
		options: {
			cwd:            { type: "string" },
			help:           { type: "boolean" },
			version:        { type: "boolean" },
			model:          { type: "string" },
			"new-session":  { type: "boolean" },
			prompt:         { type: "string" },
			thinking:       { type: "string" },
			mode:           { type: "string" },
			web:            { type: "boolean" },
			port:           { type: "string" },
		},
	});

	if (values.help) {
		printHelp();
		return;
	}

	if (values.version) {
		console.log(version ?? "unknown");
		return;
	}

	const workingDir = resolve(values.cwd ?? process.cwd());
	const { defaultThinkingLevel, launchThinkingLevel } = (() => {
		const raw = normalizeThinkingLevel(values.thinking ?? process.env.WADDINGTON_THINKING);
		return { defaultThinkingLevel: raw ?? "medium" as ThinkingLevel, launchThinkingLevel: raw };
	})();

	normalizeFeynmanSettings(settingsPath, bundledSettings, defaultThinkingLevel, authPath);

	const [command, ...rest] = positionals;

	if (command === "help") { printHelp(); return; }

	if (command === "setup") {
		await runSetup({
			settingsPath,
			bundledSettingsPath: bundledSettings,
			authPath,
			workingDir,
			sessionDir,
			appRoot,
			defaultThinkingLevel,
		});
		return;
	}

	if (command === "doctor") {
		runDoctor({ settingsPath, authPath, sessionDir, workingDir, appRoot });
		return;
	}

	if (command === "search") {
		const sub = rest[0];
		if (!sub || sub === "status") { printSearchStatus(); return; }
		if (sub === "set") {
			const providers: PiWebSearchProvider[] = ["auto", "perplexity", "exa", "gemini"];
			const p = rest[1] as PiWebSearchProvider;
			if (!p || !providers.includes(p)) throw new Error("Usage: waddington search set <auto|perplexity|exa|gemini> [key]");
			setSearchProvider(p, rest[2]);
			return;
		}
		if (sub === "clear") { clearSearchConfig(); return; }
		throw new Error(`Unknown search subcommand: ${sub}`);
	}

	if (command === "update") {
		applyFeynmanPackageManagerEnv(agentDir);
		const sources = rest[0] ? resolvePackageUpdateSources(rest[0]) : [undefined];
		for (const src of sources) {
			const r = await updateConfiguredPackages(workingDir, agentDir, src);
			r.updated.forEach(s => console.log(`Updated ${s}`));
		}
		console.log("All packages up to date.");
		return;
	}

	// Web UI mode: launch HTTP + WebSocket server instead of terminal TUI
	if (values.web) {
		const webPort = values.port ? parseInt(values.port, 10) : 3000;
		await launchWebServer({
			appRoot,
			workingDir,
			sessionDir,
			feynmanAgentDir: agentDir,
			feynmanVersion: version,
			thinkingLevel: launchThinkingLevel,
			explicitModelSpec: values.model ?? process.env.WADDINGTON_MODEL,
		}, webPort);
		return;
	}

	// Startup mode selector — only in interactive TTY without a pre-supplied prompt
	if (process.stdout.isTTY && !values.prompt && !command) {
		const choice = await select({
			message: "How would you like to start?",
			initialValue: "cli" as "cli" | "web",
			options: [
				{ value: "cli" as const, label: "Terminal  (CLI)", hint: "interactive TUI in this terminal" },
				{ value: "web" as const, label: "Web UI",          hint: "open browser at localhost:3000" },
			],
		});

		if (isCancel(choice)) { process.exit(0); }

		if (choice === "web") {
			const webPort = values.port ? parseInt(values.port, 10) : 3000;
			await launchWebServer({
				appRoot,
				workingDir,
				sessionDir,
				feynmanAgentDir: agentDir,
				feynmanVersion: version,
				thinkingLevel: launchThinkingLevel,
				explicitModelSpec: values.model ?? process.env.WADDINGTON_MODEL,
			}, webPort);
			return;
		}
	}

	// Build initial prompt from positionals (slash commands and free text)
	const mode = values.mode as "text" | "json" | "rpc" | undefined;
	let initialPrompt: string | undefined;
	let oneShotPrompt: string | undefined;

	if (values.prompt) {
		oneShotPrompt = values.prompt;
	} else if (command) {
		// If it starts with / or is a known workflow, pass as initial prompt
		const fullText = [command, ...rest].join(" ");
		initialPrompt = command.startsWith("/") ? fullText : `/${fullText}`;
		// If it doesn't look like a slash command, pass as plain text
		if (!command.startsWith("/") && !["discuss","replicate","perturb","analyze","benchmark","design","model","lit","deepresearch"].includes(command)) {
			initialPrompt = [command, ...rest].join(" ");
		}
	}

	await launchPiChat({
		appRoot,
		workingDir,
		sessionDir,
		feynmanAgentDir: agentDir,
		feynmanVersion: version,
		mode,
		thinkingLevel: launchThinkingLevel,
		explicitModelSpec: values.model ?? process.env.WADDINGTON_MODEL,
		...(oneShotPrompt  ? { oneShotPrompt }  : {}),
		...(initialPrompt  ? { initialPrompt }  : {}),
	});
}
