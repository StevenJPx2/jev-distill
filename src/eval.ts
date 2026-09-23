import { contractHash, labelsOf, type TaskContract } from "./contract.js";
import type { Dataset, DatasetRow } from "./dataset.js";
import { DistillError } from "./errors.js";
import { currentRecords, type LabelRecord, recordKey } from "./ledger.js";
import { compare, type LabelMetrics } from "./metrics.js";
import { type Prediction, createPredictor } from "./student/index.js";
import type { StudentArtifact } from "./student/artifact.js";
import { questionHash } from "./teacher/question.js";
import { teacherLabels } from "./train/index.js";

export interface Served {
  readonly coverage: number | null;
  readonly abstained: number;
  readonly metrics: LabelMetrics;
}

export interface ThresholdPoint {
  readonly threshold: number;
  readonly coverage: number | null;
  readonly rate: number | null;
}

export interface EvalReport {
  readonly report: "jev-distill.eval";
  readonly report_version: 1;
  readonly task: string;
  readonly task_version: string;
  readonly experimental: boolean;
  readonly holdout: {
    readonly rows: number;
    readonly excluded_trained_overlap: number;
    readonly teacher_labeled: number;
    readonly teacher_forbidden: number;
    readonly teacher_failed: number;
    readonly teacher_unlabeled: number;
    readonly with_gold: number;
  };
  readonly teacher_agreement: {
    readonly note: string;
    readonly argmax: LabelMetrics;
    readonly served: Served;
    readonly teacher_confident_only: LabelMetrics;
    readonly by_student_confidence: readonly ThresholdPoint[];
  };
  readonly gold: null | {
    readonly note: string;
    readonly sufficient_evidence: boolean;
    readonly student_argmax: LabelMetrics;
    readonly student_served: Served;
    readonly jev_on_same_rows: LabelMetrics;
    readonly majority_baseline: LabelMetrics & { readonly label: string | null };
    readonly by_student_confidence: readonly ThresholdPoint[];
  };
}

const AGREEMENT_NOTE =
  "Agreement with Jev on held-out rows. It measures how well the student imitates the teacher, not whether either is correct.";
const GOLD_NOTE = "Accuracy against held-out reference (gold) labels. Gold labels are never used for training or model selection.";

interface Scored {
  readonly row: DatasetRow;
  readonly prediction: Prediction;
  readonly teacher: string | null;
  readonly teacherConfidence: number | null;
}

export function evaluate(contract: TaskContract, artifact: StudentArtifact, dataset: Dataset, records: readonly LabelRecord[]): EvalReport {
  if (artifact.task.contract_hash !== contractHash(contract)) {
    throw new DistillError("invalid_artifact", "artifact was trained from a different contract; retrain or pass the matching task.json");
  }

  const labels = labelsOf(contract);
  const trained = new Set(provenanceHashes(artifact));
  const candidates = dataset.rows.filter((row) => row.split === "holdout");
  const holdout = candidates.filter((row) => !trained.has(row.contentHash));
  const current = currentRecords(records, questionHash(contract));
  const teacher = teacherLabels(contract, dataset, records);
  const predictor = createPredictor(artifact);
  const statusOf = (row: DatasetRow): string => current.get(recordKey(row.id, row.contentHash))?.status ?? "none";

  const scored: Scored[] = holdout.map((row) => {
    const label = teacher.get(row.id)?.record;
    return { row, prediction: predictor.predict(row.state), teacher: label?.choice ?? null, teacherConfidence: label?.confidence ?? null };
  });

  const withTeacher = scored.filter((s) => s.teacher !== null);
  const withGold = scored.filter((s) => s.row.gold !== null);
  const reference = (pick: (s: Scored) => string | null) => (s: Scored) => pick(s) ?? "";

  return {
    report: "jev-distill.eval",
    report_version: 1,
    task: contract.name,
    task_version: contract.version,
    experimental: contract.experimental,
    holdout: {
      rows: holdout.length,
      excluded_trained_overlap: candidates.length - holdout.length,
      teacher_labeled: withTeacher.length,
      teacher_forbidden: holdout.filter((row) => statusOf(row) === "forbidden").length,
      teacher_failed: holdout.filter((row) => statusOf(row) === "failed").length,
      teacher_unlabeled: holdout.filter((row) => statusOf(row) === "none").length,
      with_gold: withGold.length,
    },
    teacher_agreement: {
      note: AGREEMENT_NOTE,
      argmax: compare(
        labels,
        withTeacher.map((s) => ({ reference: s.teacher ?? "", predicted: s.prediction.argmax })),
      ),
      served: served(
        labels,
        withTeacher,
        reference((s) => s.teacher),
      ),
      teacher_confident_only: compare(
        labels,
        withTeacher
          .filter((s) => (s.teacherConfidence ?? 0) >= contract.uncertainty.teacher_min_confidence)
          .map((s) => ({ reference: s.teacher ?? "", predicted: s.prediction.argmax })),
      ),
      by_student_confidence: thresholds(
        contract,
        withTeacher,
        reference((s) => s.teacher),
      ),
    },
    gold: withGold.length === 0 ? null : goldSection(contract, labels, teacher, withGold),
  };
}

function goldSection(
  contract: TaskContract,
  labels: readonly string[],
  teacher: ReturnType<typeof teacherLabels>,
  rows: readonly Scored[],
): EvalReport["gold"] {
  const gold = (s: Scored): string => s.row.gold ?? "";
  // Baseline uses Jev's training labels only, so gold never informs anything but the score.
  const trainLabels = [...teacher.values()].filter(({ row }) => row.split === "train").map(({ record }) => record.choice);
  const majority = mostCommon(trainLabels, labels);

  return {
    note: GOLD_NOTE,
    sufficient_evidence: rows.length >= contract.eval.min_gold_rows,
    student_argmax: compare(
      labels,
      rows.map((s) => ({ reference: gold(s), predicted: s.prediction.argmax })),
    ),
    student_served: served(labels, rows, gold),
    jev_on_same_rows: compare(
      labels,
      rows.filter((s) => s.teacher !== null).map((s) => ({ reference: gold(s), predicted: s.teacher ?? "" })),
    ),
    majority_baseline: {
      label: majority,
      ...compare(labels, majority === null ? [] : rows.map((s) => ({ reference: gold(s), predicted: majority }))),
    },
    by_student_confidence: thresholds(contract, rows, gold),
  };
}

function served(labels: readonly string[], rows: readonly Scored[], reference: (s: Scored) => string): Served {
  const answered = rows.filter((s) => s.prediction.label !== null);

  return {
    coverage: rows.length === 0 ? null : rows.filter((s) => !s.prediction.abstained).length / rows.length,
    abstained: rows.filter((s) => s.prediction.abstained).length,
    metrics: compare(
      labels,
      answered.map((s) => ({ reference: reference(s), predicted: s.prediction.label ?? "" })),
    ),
  };
}

function thresholds(contract: TaskContract, rows: readonly Scored[], reference: (s: Scored) => string): ThresholdPoint[] {
  return contract.eval.confidence_thresholds.map((threshold) => {
    const kept = rows.filter((s) => s.prediction.confidence >= threshold);
    const correct = kept.filter((s) => s.prediction.argmax === reference(s)).length;

    return {
      threshold,
      coverage: rows.length === 0 ? null : kept.length / rows.length,
      rate: kept.length === 0 ? null : correct / kept.length,
    };
  });
}

function mostCommon(values: readonly string[], labels: readonly string[]): string | null {
  let best: string | null = null;
  let bestCount = 0;

  for (const label of labels) {
    const count = values.filter((value) => value === label).length;
    if (count > bestCount) [best, bestCount] = [label, count];
  }

  return best;
}

function provenanceHashes(artifact: StudentArtifact): string[] {
  const hashes = artifact.provenance.train_content_hashes;
  if (!Array.isArray(hashes) || !hashes.every((h) => typeof h === "string")) {
    throw new DistillError("invalid_artifact", "artifact provenance lacks train_content_hashes; cannot prove holdout isolation");
  }

  return hashes;
}
