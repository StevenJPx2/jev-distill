import { copyFileSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { Fetch } from "@typesafe-ai/sdk";

import { type Io, main } from "../src/cli.js";

export const EXAMPLES = join(import.meta.dirname, "..", "examples");

export function tempDir(prefix = "jev-distill-"): string {
  return mkdtempSync(join(tmpdir(), prefix));
}

export function readJson(path: string): Record<string, unknown> {
  return JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
}

export function patchTask(path: string, patch: (task: Record<string, any>) => void): void {
  const task = readJson(path);
  patch(task);
  writeFileSync(path, JSON.stringify(task, null, 2));
}

/** Copy an example into a temp dir, applying JSON patches to its task.json. */
export function exampleTask(
  name: "routing" | "permission",
  patch: (task: Record<string, any>) => void = () => {},
): { dir: string; task: string } {
  const dir = tempDir();
  const task = readJson(join(EXAMPLES, name, "task.json"));
  // Tests should not sleep for real backoff.
  Object.assign(task.teacher as object, { backoff_initial_ms: 1, backoff_max_ms: 5 });
  patch(task);
  writeFileSync(join(dir, "task.json"), JSON.stringify(task, null, 2));
  copyFileSync(join(EXAMPLES, name, "data.jsonl"), join(dir, "data.jsonl"));

  return { dir, task: join(dir, "task.json") };
}

export interface Run {
  readonly code: number;
  readonly stdout: string;
  readonly stderr: string;
  json(): Record<string, any>;
}

export async function cli(args: string[], extra: { fetch?: Fetch; env?: Record<string, string> } = {}): Promise<Run> {
  let stdout = "";
  let stderr = "";
  const io: Io = {
    stdout: (text) => {
      stdout += text;
    },
    stderr: (text) => {
      stderr += text;
    },
    env: extra.env ?? { TYPESAFE_API_KEY: "test-key-not-real" },
    ...(extra.fetch ? { fetch: extra.fetch } : {}),
    baseURL: "https://fake-jev.invalid",
  };
  const code = await main(args, io);

  return { code, stdout, stderr, json: () => JSON.parse(stdout) as Record<string, any> };
}
