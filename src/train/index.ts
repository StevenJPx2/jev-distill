import { contractHash, isPinnedModel, labelsOf, type TaskContract } from "../contract.js";
import type { Dataset, DatasetRow } from "../dataset.js";
import { DistillError } from "../errors.js";
import { type JsonObject, sha256Hex } from "../json.js";
import { currentRecords, type LabelRecord, recordKey } from "../ledger.js";
import { ARTIFACT_FORMAT, ARTIFACT_VERSION, type StudentArtifact } from "../student/artifact.js";
import { featureSpec, featurize } from "../student/features.js";
import { argmax, probabilities } from "../student/model.js";
import { questionHash } from "../teacher/question.js";
import { VERSION } from "../version.js";
import { fitLogistic, type TrainingSet } from "./optimizer.js";
import { selectL2 } from "./select.js";

export interface TrainOptionsCli {
  readonly allowMixedTeacher: boolean;
  readonly now?: () => string;
}

export interface TeacherLabel {
  readonly row: DatasetRow;
  readonly record: LabelRecord & {
    readonly choice: string;
    readonly probabilities: Readonly<Record<string, number>>;
    readonly confidence: number;
  };
}

/** Jev-labeled rows that match the current question and content; nothing else is a label. */
export function teacherLabels(contract: TaskContract, dataset: Dataset, records: readonly LabelRecord[]): Map<string, TeacherLabel> {
  const current = currentRecords(records, questionHash(contract));
  const out = new Map<string, TeacherLabel>();

  for (const row of dataset.rows) {
    const record = current.get(recordKey(row.id, row.contentHash));
    if (record?.status !== "labeled" || record.choice === null || record.probabilities === null || record.confidence === null) continue;

    out.set(row.id, {
      row,
      record: { ...record, choice: record.choice, probabilities: record.probabilities, confidence: record.confidence },
    });
  }

  return out;
}

export function trainStudent(
  contract: TaskContract,
  dataset: Dataset,
  records: readonly LabelRecord[],
  options: TrainOptionsCli,
): StudentArtifact {
  const labels = labelsOf(contract);
  const { used, excludedLowConfidence } = trainingLabels(contract, dataset, records);

  const answered = [...new Set(used.map(({ record }) => record.answered_model ?? "unknown"))].sort();
  checkTeacher(contract, answered, options.allowMixedTeacher);

  const spec = featureSpec(contract.train.features);
  const fieldNames = contract.state.fields.map((field) => field.name);
  const set = trainingSet(contract, used, labels, (row) => featurize(spec, fieldNames, row.state));
  const classCounts = Object.fromEntries(labels.map((label) => [label, used.filter(({ record }) => record.choice === label).length]));

  if (used.length < 4)
    throw new DistillError("training_failed", `only ${used.length} usable Jev-labeled training rows; label more rows first`);
  if (Object.values(classCounts).filter((count) => count > 0).length < 2)
    throw new DistillError("training_failed", "Jev labels cover fewer than 2 classes");

  const selection = selectL2(
    set,
    used.map(({ row }) => row.component),
    contract.train,
    contract.dataset.split_seed,
  );
  const fit = fitLogistic(set, { l2: selection.l2, epochs: contract.train.epochs, learningRate: contract.train.learning_rate });
  if (!Number.isFinite(fit.finalLoss)) throw new DistillError("training_failed", "training diverged; lower train.learning_rate");

  const trainAgreement =
    set.rows.filter((row, i) => argmax(probabilities(fit.model, row)) === argmax(set.targets[i] ?? [])).length / set.rows.length;
  const usedRecords = used
    .map(({ row, record }) => `${row.id}\u0000${row.contentHash}\u0000${record.choice}\u0000${record.request_id ?? ""}`)
    .sort();

  const provenance: JsonObject = {
    tool: `jev-distill@${VERSION}`,
    created_at: (options.now ?? (() => new Date().toISOString()))(),
    teacher: {
      requested_model: contract.teacher.model,
      answered_models: answered,
      pinned: isPinnedModel(contract.teacher.model),
      sdk: used[0]?.record.sdk ?? null,
    },
    question_hash: questionHash(contract),
    dataset_fingerprint: dataset.fingerprint,
    labels_fingerprint: sha256Hex(usedRecords.join("\n")),
    training: {
      rows: used.length,
      excluded_low_teacher_confidence: excludedLowConfidence,
      class_counts: classCounts,
      target: contract.train.target,
      class_weighting: contract.train.class_weighting,
      l2: selection.l2,
      cv: { folds: selection.folds, note: selection.note, scores: selection.scores.map((s) => ({ ...s })) },
      epochs: contract.train.epochs,
      learning_rate: contract.train.learning_rate,
      final_loss: fit.finalLoss,
      train_agreement_with_teacher: trainAgreement,
    },
    // Lets eval prove no holdout row was trained on, even if the dataset changes later.
    train_content_hashes: used.map(({ row }) => row.contentHash).sort(),
  };

  return {
    format: ARTIFACT_FORMAT,
    format_version: ARTIFACT_VERSION,
    task: artifactTask(contract),
    features: spec,
    model: fit.model,
    provenance,
  };
}

/** Training split only; the holdout and gold labels never reach the optimizer or CV. */
function trainingLabels(contract: TaskContract, dataset: Dataset, records: readonly LabelRecord[]) {
  const { teacher_min_confidence: threshold, on_low_teacher_confidence: policy } = contract.uncertainty;
  const inTrain = [...teacherLabels(contract, dataset, records).values()].filter(({ row }) => row.split === "train");
  const used = policy === "exclude" ? inTrain.filter(({ record }) => record.confidence >= threshold) : inTrain;

  return { used, excludedLowConfidence: inTrain.length - used.length };
}

function artifactTask(contract: TaskContract): StudentArtifact["task"] {
  return {
    name: contract.name,
    version: contract.version,
    contract_hash: contractHash(contract),
    experimental: contract.experimental,
    labels: labelsOf(contract),
    state_fields: contract.state.fields,
    uncertainty: {
      student_min_confidence: contract.uncertainty.student_min_confidence,
      abstain_label: contract.uncertainty.abstain_label,
    },
  };
}

function checkTeacher(contract: TaskContract, answered: readonly string[], allowMixed: boolean): void {
  if (isPinnedModel(contract.teacher.model) && answered.some((model) => model !== contract.teacher.model)) {
    throw new DistillError(
      "training_failed",
      `teacher is pinned to ${contract.teacher.model} but labels came from: ${answered.join(", ")}`,
    );
  }
  if (answered.length > 1 && !allowMixed) {
    throw new DistillError(
      "training_failed",
      `labels come from several Jev versions (${answered.join(", ")}); relabel, pin a version, or pass --allow-mixed-teacher`,
    );
  }
}

function trainingSet(
  contract: TaskContract,
  used: readonly TeacherLabel[],
  labels: readonly string[],
  features: (row: DatasetRow) => TrainingSet["rows"][number],
): TrainingSet {
  const targets = used.map(({ record }) =>
    contract.train.target === "soft"
      ? Float64Array.from(labels, (label) => record.probabilities[label] ?? 0)
      : Float64Array.from(labels, (label) => (label === record.choice ? 1 : 0)),
  );

  const counts = labels.map((_, c) => targets.filter((t) => argmax(t) === c).length);
  const present = counts.filter((count) => count > 0).length;
  const weights = Float64Array.from(targets, (t) =>
    contract.train.class_weighting === "balanced" ? used.length / (present * (counts[argmax(t)] ?? 1)) : 1,
  );

  return { rows: used.map(({ row }) => features(row)), targets, weights, labels };
}
