import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { contractHash, loadContract, parseContract } from "../src/contract.js";
import { DistillError } from "../src/errors.js";
import { questionHash } from "../src/teacher/question.js";
import { EXAMPLES, readJson } from "./helpers.js";

const routing = (): Record<string, any> => readJson(join(EXAMPLES, "routing", "task.json"));

function patched(patch: (task: Record<string, any>) => void): () => unknown {
  const task = routing();
  patch(task);
  return () => parseContract(task);
}

describe("task contract v1", () => {
  it("accepts both examples and marks the permission task experimental", () => {
    expect(loadContract(join(EXAMPLES, "routing", "task.json")).experimental).toBe(false);
    expect(loadContract(join(EXAMPLES, "permission", "task.json")).experimental).toBe(true);
  });

  it("rejects unknown fields everywhere, including an inline API key", () => {
    expect(
      patched((t) => {
        t.teacher.api_key = "sk-inline";
      }),
    ).toThrow(/\$\.teacher: unknown field\(s\): api_key/);
    expect(
      patched((t) => {
        t.decision.choices[0].lable = "x";
      }),
    ).toThrow(/unknown field\(s\): lable/);
    expect(
      patched((t) => {
        t.extra = true;
      }),
    ).toThrow(/\$: unknown field/);
  });

  it("requires a supported contract kind and version", () => {
    expect(
      patched((t) => {
        t.contract_version = 2;
      }),
    ).toThrow(/contract_version/);
    expect(
      patched((t) => {
        t.contract = "other";
      }),
    ).toThrow(/\$\.contract/);
  });

  it("allows only jev-latest or a pinned Jev version as teacher", () => {
    for (const model of ["jev-preview", "gpt-5", "laya", "jev-1.13"])
      expect(
        patched((t) => {
          t.teacher.model = model;
        }),
      ).toThrow(/\$\.teacher\.model/);
    expect(parseContract({ ...routing(), teacher: { ...routing().teacher, model: "jev-latest" } }).teacher.model).toBe("jev-latest");
  });

  it("validates the abstain policy against the declared labels", () => {
    expect(
      patched((t) => {
        t.uncertainty.abstain_label = "escalate";
      }),
    ).toThrow(/abstain_label/);
    expect(
      patched((t) => {
        t.uncertainty.student_min_confidence = 1.5;
      }),
    ).toThrow(/student_min_confidence/);
  });

  it("bounds budgets by hard ceilings", () => {
    expect(
      patched((t) => {
        t.budget.max_usd = 50;
      }),
    ).toThrow(/max_usd/);
    expect(
      patched((t) => {
        t.budget.max_requests = 1_000_000;
      }),
    ).toThrow(/max_requests/);
  });

  it("rejects duplicate labels, empty criteria, and bad feature sizes", () => {
    expect(
      patched((t) => {
        t.decision.choices[1].label = "billing";
      }),
    ).toThrow(/duplicate value "billing"/);
    expect(
      patched((t) => {
        t.decision.choices[0].criteria = 3;
      }),
    ).toThrow(/criteria/);
    expect(
      patched((t) => {
        t.train.features.dimensions = 100_000;
      }),
    ).toThrow(/power of two/);
  });

  it("changes the question hash when the question or teacher model changes, and not otherwise", () => {
    const base = parseContract(routing());
    const renamed = parseContract({ ...routing(), description: "edited description" });
    const pinned = parseContract({ ...routing(), teacher: { ...routing().teacher, model: "jev-latest" } });
    const reworded = routing();
    reworded.decision.instructions = "Which queue?";

    expect(questionHash(renamed)).toBe(questionHash(base));
    expect(contractHash(renamed)).not.toBe(contractHash(base));
    expect(questionHash(pinned)).not.toBe(questionHash(base));
    expect(questionHash(parseContract(reworded))).not.toBe(questionHash(base));
  });

  it("reports failures as typed DistillErrors", () => {
    const error = (() => {
      try {
        return parseContract({});
      } catch (caught) {
        return caught;
      }
    })();

    expect(error).toBeInstanceOf(DistillError);
    expect((error as DistillError).code).toBe("invalid_contract");
  });
});
