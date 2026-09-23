import { writeFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { loadContract, parseContract, type TaskContract } from "../src/contract.js";
import { loadDataset } from "../src/dataset.js";
import { EXAMPLES, readJson, tempDir } from "./helpers.js";

const contract = loadContract(join(EXAMPLES, "routing", "task.json"));

function dataset(lines: readonly unknown[], task: TaskContract = contract) {
  const path = join(tempDir(), "data.jsonl");
  writeFileSync(path, lines.map((line) => JSON.stringify(line)).join("\n"));

  return loadDataset(path, task);
}

const row = (subject: string, extra: Record<string, unknown> = {}) => ({ state: { subject, body: `body of ${subject}` }, ...extra });

describe("dataset ingestion", () => {
  it("derives stable content IDs independent of file order", () => {
    const a = dataset([row("one"), row("two")]);
    const b = dataset([row("two"), row("one")]);

    expect(a.rows.map((r) => r.id).sort()).toEqual(b.rows.map((r) => r.id).sort());
    expect(a.rows[0]?.id).toMatch(/^c_[0-9a-f]{20}$/);
    expect(a.fingerprint).toBe(b.fingerprint);
  });

  it("collapses case/whitespace duplicates and drops conflicting gold", () => {
    const d = dataset([row("Hello  World", { id: "x1", gold: "billing" }), row("hello world", { id: "x2", gold: "sales" })]);

    expect(d.stats).toMatchObject({ lines: 2, unique: 1, duplicates_collapsed: 1, gold_conflicts_dropped: 1 });
    expect(d.rows[0]?.gold).toBeNull();
  });

  it("keeps groups and duplicates-across-groups on one side of the split", () => {
    const lines = Array.from({ length: 200 }, (_, i) => row(`ticket ${i}`, { group: `g${i % 20}` }));
    lines.push(row("ticket 3", { group: "other-group" }));
    lines.push(row("linked", { group: "other-group" }));
    const d = dataset(lines);

    const sideByGroup = new Map<string, Set<string>>();
    for (const r of d.rows) sideByGroup.set(r.component, (sideByGroup.get(r.component) ?? new Set()).add(r.split));
    expect([...sideByGroup.values()].every((sides) => sides.size === 1)).toBe(true);

    const split = (subject: string) => d.rows.find((r) => r.state.subject === subject)?.split;
    expect(split("linked")).toBe(split("ticket 3"));
    expect(split("ticket 3")).toBe(split("ticket 23"));
  });

  it("assigns holdout deterministically from the seed", () => {
    const lines = Array.from({ length: 300 }, (_, i) => row(`t${i}`));
    const first = dataset(lines).rows.map((r) => r.split);
    const again = dataset(lines).rows.map((r) => r.split);
    const reseeded = dataset(
      lines,
      parseContract({ ...readJson(join(EXAMPLES, "routing", "task.json")), dataset: { ...contract.dataset, split_seed: "other" } }),
    );

    expect(again).toEqual(first);
    expect(reseeded.rows.map((r) => r.split)).not.toEqual(first);
    const holdout = first.filter((side) => side === "holdout").length / first.length;
    expect(holdout).toBeGreaterThan(0.2);
    expect(holdout).toBeLessThan(0.4);
  });

  it("rejects duplicate explicit IDs, undeclared fields, oversize fields, and unknown gold labels", () => {
    expect(() => dataset([row("a", { id: "same" }), row("b", { id: "same" })])).toThrow(/duplicate id/);
    expect(() => dataset([{ state: { subject: "a", body: "b", priority: "high" } }])).toThrow(/undeclared state field/);
    expect(() => dataset([{ state: { subject: "x".repeat(201), body: "b" } }])).toThrow(/max_chars/);
    expect(() => dataset([row("a", { gold: "legal" })])).toThrow(/gold/);
    expect(() => dataset([{ state: { body: "missing subject" } }])).toThrow(/missing required field "subject"/);
  });
});
