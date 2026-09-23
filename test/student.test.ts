import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { loadContract } from "../src/contract.js";
import { loadDataset } from "../src/dataset.js";
import { evaluate } from "../src/eval.js";
import { readLedger } from "../src/ledger.js";
import { artifactToJson, loadArtifact, parseArtifact } from "../src/student/artifact.js";
import { featureSpec, featurize, fnv1a32 } from "../src/student/features.js";
import { createPredictor } from "../src/student/index.js";
import { trainStudent } from "../src/train/index.js";
import { fakeJev, routingTeacher } from "./fake-jev.js";
import { cli, exampleTask, readJson } from "./helpers.js";

const SRC = join(import.meta.dirname, "..", "src");

/** Relative imports reachable from a module, by static scan. */
function importGraph(entry: string, seen = new Set<string>()): Set<string> {
  if (seen.has(entry)) return seen;
  seen.add(entry);

  for (const [, spec] of readFileSync(entry, "utf8").matchAll(/from\s+"([^"]+)"/g)) {
    if (spec?.startsWith(".")) importGraph(resolve(dirname(entry), spec.replace(/\.js$/, ".ts")), seen);
    else if (spec) seen.add(spec);
  }

  return seen;
}

async function trained() {
  const { dir, task } = exampleTask("routing");
  await cli(["label", task, "--approve"], { fetch: fakeJev({ classify: routingTeacher }).fetch });
  await cli(["train", task]);

  return { dir, task, path: join(dir, "student.json") };
}

describe("student artifact", () => {
  it("imports nothing from the Jev SDK, the teacher, or the network", () => {
    const graph = [...importGraph(join(SRC, "student", "index.ts"))];

    expect(graph.some((m) => m.includes("@typesafe-ai") || m.includes("teacher") || m.includes("label"))).toBe(false);
    expect(graph.filter((m) => !m.startsWith("/"))).toEqual(["node:fs", "node:crypto"].filter((m) => graph.includes(m)));
  });

  it("hashes features with a fixed, platform-independent function", () => {
    expect(fnv1a32("")).toBe(0x811c9dc5);
    expect(fnv1a32("a")).toBe(0xe40c292c);
    expect(fnv1a32("foobar")).toBe(0xbf9cf968);

    const spec = featureSpec({ dimensions: 1024, word_ngrams: 2, char_ngrams: null, per_field: false });
    const x = featurize(spec, ["subject"], { subject: "Refund  REFUND refund" });
    expect([...x.indices]).toEqual([...x.indices].sort((a, b) => a - b));
    expect(Math.hypot(...x.values)).toBeCloseTo(1, 12);
  });

  it("rejects tampered artifacts and round-trips weights exactly", async () => {
    const { path } = await trained();
    const artifact = loadArtifact(path);
    const again = parseArtifact(JSON.parse(JSON.stringify(artifactToJson(artifact))));
    expect([...again.model.weights]).toEqual([...artifact.model.weights]);

    const raw = readJson(path) as Record<string, any>;
    raw.model.bias[0] += 1;
    writeFileSync(path, JSON.stringify(raw));
    expect(() => loadArtifact(path)).toThrow(/digest mismatch/);
  });

  it("applies the abstain policy with the served label", async () => {
    const { path } = await trained();
    const artifact = loadArtifact(path);
    const strict = createPredictor({
      ...artifact,
      task: { ...artifact.task, uncertainty: { student_min_confidence: 1, abstain_label: null } },
    });
    const lenient = createPredictor({
      ...artifact,
      task: { ...artifact.task, uncertainty: { student_min_confidence: 0, abstain_label: null } },
    });
    const state = { subject: "hello", body: "question about something" };

    expect(strict.predict(state)).toMatchObject({ abstained: true, label: null });
    expect(lenient.predict(state).abstained).toBe(false);
    expect(lenient.predict(state).label).toBe(lenient.predict(state).argmax);
  });

  it("never trains on holdout or gold, and eval excludes rows proven trained-on", async () => {
    const { dir, task } = await trained();
    const contract = loadContract(task);
    const data = loadDataset(join(dir, "data.jsonl"), contract);
    const artifact = trainStudent(contract, data, readLedger(join(dir, "labels.jsonl")), { allowMixedTeacher: false });
    const trainedHashes = new Set(artifact.provenance.train_content_hashes as string[]);

    expect(data.rows.filter((r) => r.split === "holdout").some((r) => trainedHashes.has(r.contentHash))).toBe(false);

    // Simulate a leak: pretend one holdout row was trained on.
    const leaked = data.rows.find((r) => r.split === "holdout");
    const leakyArtifact = {
      ...artifact,
      provenance: { ...artifact.provenance, train_content_hashes: [...trainedHashes, leaked?.contentHash ?? ""] },
    };
    const report = evaluate(contract, leakyArtifact, data, readLedger(join(dir, "labels.jsonl")));
    expect(report.holdout.excluded_trained_overlap).toBe(1);
  });

  it("refuses to train on labels from an unpinned mix of Jev versions", async () => {
    const { dir, task } = exampleTask("routing", (t) => {
      t.teacher.model = "jev-latest";
    });
    await cli(["label", task, "--approve", "--max-requests", "50"], {
      fetch: fakeJev({ classify: routingTeacher, model: "jev-1.13.0" }).fetch,
    });
    await cli(["label", task, "--approve"], { fetch: fakeJev({ classify: routingTeacher, model: "jev-1.14.0" }).fetch });

    const refused = await cli(["train", task]);
    expect(refused.code).toBe(1);
    expect(refused.stderr).toContain("several Jev versions");
    expect((await cli(["train", task, "--allow-mixed-teacher"])).code).toBe(0);
    expect(readJson(join(dir, "student.json")).provenance).toMatchObject({
      teacher: { answered_models: ["jev-1.13.0", "jev-1.14.0"], pinned: false },
    });
  });
});
