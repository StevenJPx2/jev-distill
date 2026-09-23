import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { loadContract } from "../src/contract.js";
import { loadDataset } from "../src/dataset.js";
import { planLabeling, runLabeling } from "../src/label.js";
import { readLedger } from "../src/ledger.js";
import { createJevTeacher } from "../src/teacher/jev.js";
import { fakeJev, type FakeJevOptions, routingTeacher } from "./fake-jev.js";
import { cli, exampleTask } from "./helpers.js";

function setup(patch: (task: Record<string, any>) => void = () => {}) {
  const { dir, task } = exampleTask("routing", patch);
  const contract = loadContract(task);
  const data = loadDataset(join(dir, "data.jsonl"), contract);
  const ledger = join(dir, "labels.jsonl");

  const label = async (options: Partial<FakeJevOptions>, plan = { retryFailed: false, overrides: {} }) => {
    const jev = fakeJev({ classify: routingTeacher, ...options });
    const teacher = createJevTeacher(contract, { apiKey: "k", fetch: jev.fetch, baseURL: "https://fake-jev.invalid" });
    const summary = await runLabeling(contract, planLabeling(contract, data, ledger, plan), ledger, { approve: true, teacher });

    return { summary, jev, records: readLedger(ledger) };
  };

  return { dir, task, contract, data, ledger, label };
}

describe("bounded Jev labeling", () => {
  it("refuses to spend without approval and a dry run sends nothing", async () => {
    const { contract, data, ledger, task } = setup();
    const jev = fakeJev({ classify: routingTeacher });
    const teacher = createJevTeacher(contract, { apiKey: "k", fetch: jev.fetch });
    const plan = planLabeling(contract, data, ledger, { retryFailed: false, overrides: {} });

    await expect(runLabeling(contract, plan, ledger, { approve: false, teacher })).rejects.toThrow(/--approve/);
    expect((await cli(["label", task, "--approve", "--dry-run"], { fetch: jev.fetch })).json().dry_run).toBe(true);
    expect(jev.requests).toHaveLength(0);
    expect(plan.planned_worst_case.usd).toBeLessThanOrEqual(contract.budget.max_usd);
  });

  it("requires the API key from the environment variable named in the contract", async () => {
    const { task } = setup();
    const run = await cli(["label", task, "--approve"], { env: {} });

    expect(run.code).toBe(1);
    expect(run.stderr).toContain("set TYPESAFE_API_KEY");
  });

  it("enforces request budgets cumulatively across runs, and CLI limits can only lower them", async () => {
    const { label, contract, data, ledger } = setup((t) => {
      t.budget.max_requests = 30;
    });

    const first = await label({}, { retryFailed: false, overrides: { max_requests: 10 } });
    expect(first.summary.attempted).toBe(10);
    const raised = await label({}, { retryFailed: false, overrides: { max_requests: 500 } });
    expect(raised.summary.attempted).toBe(20);
    expect((await label({})).summary.attempted).toBe(0);

    const plan = planLabeling(contract, data, ledger, { retryFailed: false, overrides: {} });
    expect(plan).toMatchObject({ limited_by: "max_requests", selected: 0, already_labeled: 30 });
  });

  it("stops before exceeding the dollar budget using worst-case estimates", async () => {
    const { contract, data, ledger } = setup((t) => {
      t.budget.max_usd = 0.0001;
    });
    const plan = planLabeling(contract, data, ledger, { retryFailed: false, overrides: {} });

    expect(plan.limited_by).toBe("max_usd");
    expect(plan.spent_before.usd + plan.planned_worst_case.usd).toBeLessThanOrEqual(0.0001);
  });

  it("marks content-specific 403s forbidden without a label and keeps going", async () => {
    const { label } = setup();
    const { summary, records } = await label({ forbid: (text) => text.includes("refund") });

    const forbidden = records.filter((r) => r.status === "forbidden");
    expect(summary.stopped).toBe("completed");
    expect(summary.forbidden).toBe(forbidden.length);
    expect(forbidden.length).toBeGreaterThan(1);
    expect(forbidden.every((r) => r.choice === null && r.probabilities === null && r.attempts === 1 && r.error?.status === 403)).toBe(true);
    expect(summary.labeled).toBe(100 - forbidden.length);
  });

  it("retries transient failures within the bounded retry policy and records attempts", async () => {
    const { label } = setup((t) => {
      t.teacher.max_retries = 2;
    });
    const { summary, records } = await label({
      flaky: (text, attempt) => (text.includes("Webhook") && attempt < 2 ? 429 : text.includes("CSV") ? 503 : null),
    });

    const webhook = records.find((r) => r.id === "t03");
    const csv = records.find((r) => r.id === "t05");
    expect(webhook).toMatchObject({ status: "labeled", attempts: 3 });
    expect(csv).toMatchObject({ status: "failed", attempts: 3, choice: null, error: { reason: "server_error", status: 503 } });
    expect(summary.failed).toBe(1);
  });

  it("stops after too many consecutive failures and on fatal auth errors", async () => {
    const consecutive = await setup((t) => {
      t.budget.max_consecutive_failures = 3;
      t.teacher.concurrency = 1;
    }).label({ forbid: () => true });
    expect(consecutive.summary).toMatchObject({ stopped: "consecutive_failures", forbidden: 3, labeled: 0 });

    const fatal = await setup().label({ always: 401 });
    expect(fatal.summary).toMatchObject({ stopped: "fatal", attempted: 0 });
    expect(fatal.records).toHaveLength(0);
  });

  it("never turns a malformed answer into a label", async () => {
    const { label } = setup();
    const { records } = await label({ malformed: (text) => text.includes("Refund request") });

    expect(records.find((r) => r.id === "b02")).toMatchObject({ status: "failed", choice: null, error: { reason: "invalid_response" } });
  });

  it("retries failed rows only when asked, and never stores state text or the key", async () => {
    const { label, ledger } = setup();
    await label({ forbid: (text) => text.includes("GDPR") });
    expect((await label({})).summary.attempted).toBe(0);

    const retry = await label({}, { retryFailed: true, overrides: {} });
    expect(retry.summary).toMatchObject({ attempted: 1, labeled: 1 });

    const text = readFileSync(ledger, "utf8");
    expect(text).not.toMatch(/GDPR|personal data|"k"/);
    expect(readLedger(ledger).every((r) => r.sdk === "@typesafe-ai/sdk@0.6.0" && r.requested_model === "jev-1.13.0")).toBe(true);
  });

  it("records Jev probabilities, confidence, answering model, usage, and request IDs", async () => {
    const { records } = await setup().label({});
    const record = records.find((r) => r.status === "labeled");

    expect(record?.request_id).toMatch(/^req_200_/);
    expect(record?.answered_model).toBe("jev-1.13.0");
    expect(Object.values(record?.probabilities ?? {}).reduce((s, p) => s + p, 0)).toBeCloseTo(1, 6);
    expect(record?.confidence).toBeGreaterThanOrEqual(0);
    expect(record?.charged_input_tokens).toBe(record?.usage?.input_tokens);
  });
});
