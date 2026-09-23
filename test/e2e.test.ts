import { copyFileSync, readFileSync } from "node:fs";
import { request } from "node:http";
import type { AddressInfo } from "node:net";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { createStudentServer } from "../src/server.js";
import { loadPredictor } from "../src/student/index.js";
import { fakeJev, routingTeacher } from "./fake-jev.js";
import { cli, exampleTask, patchTask, readJson, tempDir } from "./helpers.js";

describe("routing example, end to end with a fake Jev transport", () => {
  it("inits, validates, dry-runs, labels, trains, evaluates, predicts, and serves", async () => {
    const dir = tempDir();
    const init = await cli(["task", "init", dir, "--template", "routing"]);
    expect(init.code).toBe(0);
    expect((await cli(["task", "init", dir])).code).toBe(1);

    const task = join(dir, "task.json");
    patchTask(task, (t) => Object.assign(t.teacher, { backoff_initial_ms: 1, backoff_max_ms: 5 }));
    const validate = await cli(["task", "validate", task, "--data", join(dir, "data.jsonl")]);
    expect(validate.code).toBe(0);
    expect(validate.json().dataset).toMatchObject({ lines: 102, unique: 100, duplicates_collapsed: 2 });

    const jev = fakeJev({ classify: routingTeacher, forbid: (text) => text.includes("GDPR") });
    const dry = await cli(["label", task], { fetch: jev.fetch });
    expect(dry.code).toBe(0);
    expect(dry.json().dry_run).toBe(true);
    expect(dry.json().plan).toMatchObject({ rows: 100, pending: 100, selected: 100, limited_by: "none", teacher_model: "jev-1.13.0" });
    expect(jev.requests).toHaveLength(0);

    const label = await cli(["label", task, "--approve"], { fetch: jev.fetch });
    expect(label.code).toBe(0);
    const run = label.json().run;
    expect(run).toMatchObject({ attempted: 100, forbidden: 1, stopped: "completed", answered_models: ["jev-1.13.0"] });
    expect(run.labeled).toBe(99);
    expect(jev.requests.every((r) => r.authorization === "Bearer test-key-not-real" && r.body.model === "jev-1.13.0")).toBe(true);

    const ledger = readFileSync(join(dir, "labels.jsonl"), "utf8");
    expect(ledger).not.toContain("Charged twice");
    expect(ledger).not.toContain("test-key-not-real");

    const train = await cli(["train", task]);
    expect(train.code).toBe(0);
    expect(train.json().training.rows).toBeGreaterThan(50);

    const evaluation = await cli(["eval", task]);
    expect(evaluation.code).toBe(0);
    expect(evaluation.stdout).toContain("Agreement with Jev (teacher imitation, not accuracy)");
    expect(evaluation.stdout).toContain("Accuracy vs held-out gold");

    const report = readJson(join(dir, "report.json")) as Record<string, any>;
    expect(report.holdout.rows).toBeGreaterThan(15);
    expect(report.holdout.excluded_trained_overlap).toBe(0);
    expect(report.teacher_agreement.argmax.n).toBe(report.holdout.teacher_labeled);
    expect(report.gold.student_argmax.n).toBe(report.holdout.with_gold);
    // The fake teacher is imperfect, so imitation and correctness are measured separately and differ.
    expect(report.gold.jev_on_same_rows.rate).toBeLessThan(1);
    expect(report.teacher_agreement.argmax.rate).toBeGreaterThan(report.gold.majority_baseline.rate);

    const predict = await cli([
      "predict",
      join(dir, "student.json"),
      "--state",
      JSON.stringify({ subject: "Refund please", body: "I was charged twice for my invoice" }),
    ]);
    expect(predict.code).toBe(0);
    expect(predict.json()).toMatchObject({ task: "support-routing", argmax: "billing", experimental: false });

    const bad = await cli(["predict", join(dir, "student.json"), "--state", JSON.stringify({ subject: "x", body: "y", extra: 1 })]);
    expect(bad.code).toBe(1);
    expect(bad.stderr).toContain("undeclared state field");
  });

  it("produces a portable, deterministic artifact that predicts with no teacher configured", async () => {
    const { task, dir } = exampleTask("routing");
    const jev = fakeJev({ classify: routingTeacher });
    await cli(["label", task, "--approve"], { fetch: jev.fetch });

    await cli(["train", task, "--out", join(dir, "a.json")]);
    await cli(["train", task, "--out", join(dir, "b.json")]);
    const [a, b] = [readJson(join(dir, "a.json")), readJson(join(dir, "b.json"))];
    expect(a.model).toEqual(b.model);

    // "Another machine": copy only the artifact somewhere else and load it through the student entry point.
    const elsewhere = join(tempDir(), "student.json");
    copyFileSync(join(dir, "a.json"), elsewhere);
    const predictor = loadPredictor(elsewhere);
    const prediction = predictor.predict({
      subject: "Cannot log in",
      body: "My password reset email never arrives and I am locked out of my account",
    });
    expect(prediction.argmax).toBe("account");
    expect(Object.values(prediction.probabilities).reduce((s, p) => s + p, 0)).toBeCloseTo(1, 9);
  });

  it("serves predictions over local HTTP", async () => {
    const { task, dir } = exampleTask("routing");
    await cli(["label", task, "--approve"], { fetch: fakeJev({ classify: routingTeacher }).fetch });
    await cli(["train", task]);

    const predictor = loadPredictor(join(dir, "student.json"));
    const server = createStudentServer(new Map([["support-routing", predictor]]));
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const { port } = server.address() as AddressInfo;

    const post = (path: string, body: string) =>
      new Promise<{ status: number; body: Record<string, any> }>((resolve, reject) => {
        const req = request({ host: "127.0.0.1", port, path, method: "POST", headers: { "content-type": "application/json" } }, (res) => {
          let text = "";
          res.on("data", (chunk: Buffer) => {
            text += chunk.toString();
          });
          res.on("end", () => resolve({ status: res.statusCode ?? 0, body: JSON.parse(text) as Record<string, any> }));
        });
        req.on("error", reject);
        req.end(body);
      });

    try {
      const ok = await post(
        "/v1/predict/support-routing",
        JSON.stringify({ state: { subject: "Demo", body: "Can we book a pricing demo for 300 seats?" } }),
      );
      expect(ok.status).toBe(200);
      expect(ok.body.argmax).toBe("sales");

      const invalid = await post("/v1/predict/support-routing", JSON.stringify({ state: { subject: 3 } }));
      expect(invalid.status).toBe(400);
      expect(invalid.body.error).toBe("invalid_state");

      const missing = await post("/v1/predict/nope", "{}");
      expect(missing.status).toBe(404);
    } finally {
      await new Promise((resolve) => server.close(resolve));
    }
  });
});
