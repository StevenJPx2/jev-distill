import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { main } from "../src/cli.js";
import { readLedger } from "../src/ledger.js";
import { exampleTask } from "./helpers.js";

/**
 * Opt-in live check against the real Jev API. Sends at most 6 small requests
 * (about 3k input tokens, well under $0.001 at the documented price).
 * Run with: JEV_DISTILL_LIVE=1 TYPESAFE_API_KEY=... pnpm smoke:live
 */
const enabled = process.env.JEV_DISTILL_LIVE === "1" && Boolean(process.env.TYPESAFE_API_KEY?.trim());

describe.skipIf(!enabled)("live Jev smoke (opt-in)", () => {
  it("labels a handful of rows with the real SDK and records provenance", async () => {
    const { dir, task } = exampleTask("routing", (t) => {
      Object.assign(t.teacher, { backoff_initial_ms: 500, backoff_max_ms: 5000, concurrency: 1 });
      Object.assign(t.budget, { max_requests: 6, max_input_tokens: 6000, max_usd: 0.001 });
    });
    let stdout = "";
    const code = await main(["label", task, "--approve"], {
      stdout: (text) => {
        stdout += text;
      },
      stderr: (text) => process.stderr.write(text),
      env: process.env,
    });
    const result = JSON.parse(stdout) as { run: { attempted: number; labeled: number; spent_this_run: { usd: number } } };
    const records = readLedger(join(dir, "labels.jsonl"));

    console.log(
      JSON.stringify({
        run: result.run,
        statuses: records.map((r) => [r.id, r.status, r.choice, r.answered_model, r.usage?.input_tokens]),
      }),
    );
    expect(code).toBe(0);
    expect(result.run.attempted).toBeLessThanOrEqual(6);
    expect(result.run.spent_this_run.usd).toBeLessThanOrEqual(0.001);
    expect(records.length).toBe(result.run.attempted);
    const labeled = records.filter((r) => r.status === "labeled");
    expect(labeled.every((r) => r.answered_model === "jev-1.13.0" && (r.usage?.input_tokens ?? 0) > 0)).toBe(true);
  });
});
