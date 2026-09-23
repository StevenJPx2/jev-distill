#!/usr/bin/env node
import { copyFileSync, existsSync, mkdirSync, realpathSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import type { Fetch } from "@typesafe-ai/sdk";

import { contractHash, loadContract, type TaskContract } from "./contract.js";
import { loadDataset, readJsonLines } from "./dataset.js";
import { DistillError } from "./errors.js";
import { evaluate } from "./eval.js";
import { type LabelPlan, planLabeling, runLabeling } from "./label.js";
import { readLedger } from "./ledger.js";
import { renderEval } from "./render.js";
import { createStudentServer } from "./server.js";
import { artifactToJson } from "./student/artifact.js";
import { loadPredictor } from "./student/index.js";
import { createJevTeacher } from "./teacher/jev.js";
import { trainStudent } from "./train/index.js";
import { VERSION } from "./version.js";

export interface Io {
  readonly stdout: (text: string) => void;
  readonly stderr: (text: string) => void;
  readonly env: Readonly<Record<string, string | undefined>>;
  /** Transport seam for offline tests; the real CLI uses global fetch. */
  readonly fetch?: Fetch;
  readonly baseURL?: string;
}

const TEMPLATES = ["routing", "permission"] as const;
const USAGE = `jev-distill ${VERSION}

  task init <dir> [--template routing|permission] [--force]
  task validate <task.json> [--data data.jsonl]
  label <task.json> [--data F] [--labels F] [--approve] [--retry-failed]
                    [--max-requests N] [--max-input-tokens N] [--max-usd X]
  train <task.json> [--data F] [--labels F] [--out student.json] [--allow-mixed-teacher]
  eval  <task.json> [--artifact student.json] [--data F] [--labels F] [--out report.json]
  predict <student.json> (--state '<json>' | --input rows.jsonl)
  serve <student.json>... [--host 127.0.0.1] [--port 8787]

label is a dry run unless --approve is given. Files default to the task.json directory.`;

const OPTIONS = {
  data: { type: "string" },
  labels: { type: "string" },
  out: { type: "string" },
  artifact: { type: "string" },
  template: { type: "string" },
  state: { type: "string" },
  input: { type: "string" },
  host: { type: "string" },
  port: { type: "string" },
  "max-requests": { type: "string" },
  "max-input-tokens": { type: "string" },
  "max-usd": { type: "string" },
  approve: { type: "boolean" },
  "dry-run": { type: "boolean" },
  "retry-failed": { type: "boolean" },
  "allow-mixed-teacher": { type: "boolean" },
  force: { type: "boolean" },
  help: { type: "boolean" },
} as const;

type Parsed = ReturnType<typeof parseArgs<{ options: typeof OPTIONS; allowPositionals: true }>>;
type Values = Parsed["values"];

export async function main(argv: readonly string[], io: Io): Promise<number> {
  let parsed: Parsed;
  try {
    parsed = parseArgs({ args: [...argv], options: OPTIONS, allowPositionals: true, strict: true });
  } catch (error) {
    io.stderr(`${(error as Error).message}\n\n${USAGE}\n`);
    return 2;
  }

  const [command, ...rest] = parsed.positionals;
  if (parsed.values.help === true || command === undefined) {
    io.stdout(`${USAGE}\n`);
    return command === undefined && parsed.values.help !== true ? 2 : 0;
  }

  try {
    return await dispatch(command, rest, parsed.values, io);
  } catch (error) {
    if (!(error instanceof DistillError)) throw error;
    io.stderr(`error [${error.code}]: ${error.message}\n`);
    return error.code === "usage" ? 2 : 1;
  }
}

async function dispatch(command: string, args: readonly string[], values: Values, io: Io): Promise<number> {
  const first = (): string => args[0] ?? usage(`${command} needs a path argument`);

  switch (command) {
    case "task":
      return taskCommand(args, values, io);
    case "label":
      return labelCommand(first(), values, io);
    case "train":
      return trainCommand(first(), values, io);
    case "eval":
      return evalCommand(first(), values, io);
    case "predict":
      return predictCommand(first(), values, io);
    case "serve":
      return serveCommand(args, values, io);
    default:
      return usage(`unknown command "${command}"`);
  }
}

function usage(message: string): never {
  throw new DistillError("usage", `${message}\n\n${USAGE}`);
}

function paths(taskPath: string, values: Values) {
  const dir = dirname(resolve(taskPath));

  return {
    data: values.data ?? join(dir, "data.jsonl"),
    labels: values.labels ?? join(dir, "labels.jsonl"),
    artifact: values.artifact ?? values.out ?? join(dir, "student.json"),
    report: values.out ?? join(dir, "report.json"),
  };
}

function loadTask(path: string, io: Io): TaskContract {
  const contract = loadContract(path);
  if (contract.experimental) io.stderr(`warning: task "${contract.name}" is marked EXPERIMENTAL; do not use it for real decisions.\n`);

  return contract;
}

function print(io: Io, value: unknown): void {
  io.stdout(`${JSON.stringify(value, null, 2)}\n`);
}

function taskCommand(args: readonly string[], values: Values, io: Io): number {
  const [sub, target] = args;
  if (target === undefined) return usage("task needs a subcommand and a path");

  if (sub === "validate") {
    const contract = loadTask(target, io);
    const summary: Record<string, unknown> = {
      valid: true,
      name: contract.name,
      version: contract.version,
      contract_hash: contractHash(contract),
    };
    if (values.data !== undefined) summary.dataset = loadDataset(values.data, contract).stats;
    print(io, summary);
    return 0;
  }

  if (sub !== "init") return usage(`unknown task subcommand "${sub ?? ""}"`);

  const template =
    TEMPLATES.find((name) => name === (values.template ?? "routing")) ?? usage(`--template must be one of ${TEMPLATES.join(", ")}`);
  const source = fileURLToPath(new URL(`../examples/${template}/`, import.meta.url));
  const files = ["task.json", "data.jsonl"];

  mkdirSync(target, { recursive: true });
  for (const file of files) {
    if (existsSync(join(target, file)) && values.force !== true)
      throw new DistillError("io", `${join(target, file)} exists; pass --force to overwrite`);
    copyFileSync(join(source, file), join(target, file));
  }

  print(io, { created: files.map((file) => join(target, file)), template, next: `jev-distill label ${join(target, "task.json")}` });
  return 0;
}

function limitOverrides(values: Values) {
  const num = (name: "max-requests" | "max-input-tokens" | "max-usd"): number | undefined => {
    const raw = values[name];
    if (raw === undefined) return undefined;
    const value = Number(raw);
    return Number.isFinite(value) && value >= 0 ? value : usage(`--${name} must be a non-negative number`);
  };
  const out: { max_requests?: number; max_input_tokens?: number; max_usd?: number } = {};
  const [requests, tokens, dollars] = [num("max-requests"), num("max-input-tokens"), num("max-usd")];
  if (requests !== undefined) out.max_requests = requests;
  if (tokens !== undefined) out.max_input_tokens = tokens;
  if (dollars !== undefined) out.max_usd = dollars;

  return out;
}

function planView(plan: LabelPlan): Omit<LabelPlan, "queue"> {
  const { queue: _queue, ...view } = plan;
  return view;
}

async function labelCommand(taskPath: string, values: Values, io: Io): Promise<number> {
  const contract = loadTask(taskPath, io);
  const files = paths(taskPath, values);
  const dataset = loadDataset(files.data, contract);
  const plan = planLabeling(contract, dataset, files.labels, {
    retryFailed: values["retry-failed"] === true,
    overrides: limitOverrides(values),
  });

  if (values.approve !== true || values["dry-run"] === true) {
    print(io, { dry_run: true, plan: planView(plan), approve_with: `jev-distill label ${taskPath} --approve` });
    return 0;
  }

  const apiKey = io.env[contract.teacher.api_key_env]?.trim();
  if (!apiKey) throw new DistillError("teacher_unavailable", `set ${contract.teacher.api_key_env} to label with Jev`);

  const teacher = createJevTeacher(contract, {
    apiKey,
    ...(io.fetch ? { fetch: io.fetch } : {}),
    ...(io.baseURL ? { baseURL: io.baseURL } : {}),
  });
  const summary = await runLabeling(contract, plan, files.labels, { approve: true, teacher });
  print(io, { dry_run: false, plan: planView(plan), run: summary, labels: files.labels });

  return summary.stopped === "completed" || summary.stopped === "budget" ? 0 : 1;
}

function trainCommand(taskPath: string, values: Values, io: Io): number {
  const contract = loadTask(taskPath, io);
  const files = paths(taskPath, values);
  const dataset = loadDataset(files.data, contract);
  const artifact = trainStudent(contract, dataset, readLedger(files.labels), { allowMixedTeacher: values["allow-mixed-teacher"] === true });
  const out = values.out ?? files.artifact;

  writeFileSync(out, `${JSON.stringify(artifactToJson(artifact))}\n`);
  print(io, { artifact: out, dataset: dataset.stats, training: artifact.provenance.training });
  return 0;
}

function evalCommand(taskPath: string, values: Values, io: Io): number {
  const contract = loadTask(taskPath, io);
  const { out: _out, ...rest } = values;
  const files = paths(taskPath, rest);
  const predictor = loadPredictor(files.artifact);
  const report = evaluate(contract, predictor.artifact, loadDataset(files.data, contract), readLedger(files.labels));
  const out = values.out ?? files.report;

  writeFileSync(out, `${JSON.stringify(report, null, 2)}\n`);
  io.stdout(renderEval(report));
  io.stderr(`report written to ${out}\n`);
  return 0;
}

function predictCommand(artifactPath: string, values: Values, io: Io): number {
  const predictor = loadPredictor(artifactPath);

  if (values.state !== undefined) {
    let state: unknown;
    try {
      state = JSON.parse(values.state);
    } catch {
      return usage("--state must be JSON");
    }
    print(io, predictor.predict(state));
    return 0;
  }

  if (values.input === undefined) return usage("predict needs --state or --input");

  for (const { line, value } of readJsonLines(values.input, 1_048_576, 1_000_000)) {
    const state = typeof value === "object" && value !== null && "state" in value ? (value as { state: unknown }).state : value;
    const id = typeof value === "object" && value !== null && "id" in value ? (value as { id: unknown }).id : line;
    io.stdout(`${JSON.stringify({ id, ...predictor.predict(state) })}\n`);
  }

  return 0;
}

async function serveCommand(args: readonly string[], values: Values, io: Io): Promise<number> {
  if (args.length === 0) return usage("serve needs at least one student artifact");

  const predictors = new Map(
    args.map((path) => {
      const predictor = loadPredictor(path);
      return [predictor.artifact.task.name, predictor] as const;
    }),
  );
  const port = Number(values.port ?? "8787");
  if (!Number.isInteger(port) || port < 0 || port > 65_535) return usage("--port must be an integer 0-65535");

  const host = values.host ?? "127.0.0.1";
  const server = createStudentServer(predictors);
  await new Promise<void>((resolveListen) => server.listen(port, host, resolveListen));
  io.stderr(`serving ${[...predictors.keys()].join(", ")} on http://${host}:${port} (POST /v1/predict/<task>, GET /healthz)\n`);

  await new Promise<void>((resolveClose) => server.on("close", resolveClose));
  return 0;
}

function isEntryPoint(): boolean {
  const entry = process.argv[1];
  if (entry === undefined) return false;

  try {
    return realpathSync(entry) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (isEntryPoint()) {
  const code = await main(process.argv.slice(2), {
    stdout: (text) => process.stdout.write(text),
    stderr: (text) => process.stderr.write(text),
    env: process.env,
  });
  process.exitCode = code;
}
