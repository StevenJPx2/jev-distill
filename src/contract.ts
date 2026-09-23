import { readFileSync } from "node:fs";

import { DistillError } from "./errors.js";
import { canonicalJson, Decoder, type Entry, type JsonObject, parseJson, sha256Hex } from "./json.js";

export const CONTRACT_KIND = "jev-distill.task";
export const CONTRACT_VERSION = 1;

/** Hard ceilings independent of any contract, so a typo cannot authorize a large spend. */
export const HARD_LIMITS = { maxRequests: 50_000, maxUsd: 10, maxInputTokens: 500_000_000 } as const;

export type FieldType = "string" | "number" | "boolean" | "json";

export interface StateField {
  readonly name: string;
  readonly type: FieldType;
  readonly description: string;
  readonly required: boolean;
  readonly max_chars: number;
}

export interface Choice {
  readonly label: string;
  readonly description: string;
  /** Rubric sent to Jev for this option; defaults to `description`. */
  readonly criteria: Entry;
}

export interface UncertaintyPolicy {
  /** Jev answers below this confidence are "low confidence". */
  readonly teacher_min_confidence: number;
  readonly on_low_teacher_confidence: "exclude" | "keep";
  /** The student abstains when its confidence is below this. */
  readonly student_min_confidence: number;
  /** Served label when abstaining; `null` means "no label" (caller decides). */
  readonly abstain_label: string | null;
}

export interface Pricing {
  readonly input_usd_per_mtok: number;
  readonly output_usd_per_mtok: number;
  readonly source: string;
  readonly checked_on: string;
}

export interface TeacherSpec {
  /** `jev-latest` (moving alias) or a pinned version such as `jev-1.13.0`. */
  readonly model: string;
  readonly api_key_env: string;
  readonly timeout_ms: number;
  readonly max_retries: number;
  readonly backoff_initial_ms: number;
  readonly backoff_max_ms: number;
  readonly concurrency: number;
  readonly pricing: Pricing;
}

export interface Budget {
  readonly max_requests: number;
  readonly max_input_tokens: number;
  readonly max_usd: number;
  readonly max_consecutive_failures: number;
}

export interface DatasetOptions {
  readonly holdout_fraction: number;
  readonly split_seed: string;
  readonly max_rows: number;
  readonly max_line_bytes: number;
}

export interface FeatureOptions {
  /** Hash space size; a power of two. */
  readonly dimensions: number;
  readonly word_ngrams: number;
  readonly char_ngrams: readonly [number, number] | null;
  readonly per_field: boolean;
}

export interface TrainOptions {
  readonly features: FeatureOptions;
  readonly target: "soft" | "hard";
  readonly class_weighting: "none" | "balanced";
  readonly l2_grid: readonly number[];
  readonly cv_folds: number;
  readonly epochs: number;
  readonly learning_rate: number;
}

export interface EvalOptions {
  readonly confidence_thresholds: readonly number[];
  /** Gold metrics on fewer rows than this are flagged as insufficient evidence. */
  readonly min_gold_rows: number;
}

export interface TaskContract {
  readonly contract: typeof CONTRACT_KIND;
  readonly contract_version: typeof CONTRACT_VERSION;
  readonly name: string;
  readonly version: string;
  readonly description: string;
  readonly experimental: boolean;
  readonly state: { readonly fields: readonly StateField[] };
  readonly decision: { readonly instructions: Entry; readonly choices: readonly Choice[] };
  readonly uncertainty: UncertaintyPolicy;
  readonly teacher: TeacherSpec;
  readonly budget: Budget;
  readonly dataset: DatasetOptions;
  readonly train: TrainOptions;
  readonly eval: EvalOptions;
}

const d = new Decoder("invalid_contract");
const NAME = /^[a-z][a-z0-9_-]{0,63}$/;
const TEACHER_MODEL = /^jev-(latest|\d+\.\d+\.\d+)$/;

export function labelsOf(contract: TaskContract): string[] {
  return contract.decision.choices.map((choice) => choice.label);
}

export function isPinnedModel(model: string): boolean {
  return model !== "jev-latest";
}

export function contractHash(contract: TaskContract): string {
  return sha256Hex(canonicalJson(contract as unknown as JsonObject));
}

export function loadContract(path: string): TaskContract {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    throw new DistillError("io", `cannot read contract ${path}: ${(error as Error).message}`);
  }

  return parseContract(parseJson(text, "invalid_contract", path));
}

export function parseContract(raw: unknown): TaskContract {
  const o = d.object(raw, "$", [
    "contract",
    "contract_version",
    "name",
    "version",
    "description",
    "experimental",
    "state",
    "decision",
    "uncertainty",
    "teacher",
    "budget",
    "dataset",
    "train",
    "eval",
  ]);

  if (o.contract !== CONTRACT_KIND) d.fail("$.contract", `must be "${CONTRACT_KIND}"`);
  if (o.contract_version !== CONTRACT_VERSION) d.fail("$.contract_version", `unsupported; this build reads version ${CONTRACT_VERSION}`);

  const decision = parseDecision(o.decision);
  const labels = decision.choices.map((choice) => choice.label);

  return {
    contract: CONTRACT_KIND,
    contract_version: CONTRACT_VERSION,
    name: d.string(o.name, "$.name", { pattern: NAME }),
    version: d.string(o.version, "$.version", { pattern: /^\d+\.\d+\.\d+$/ }),
    description: d.string(o.description, "$.description", { max: 2000 }),
    experimental: o.experimental === undefined ? false : d.boolean(o.experimental, "$.experimental"),
    state: parseState(o.state),
    decision,
    uncertainty: parseUncertainty(o.uncertainty, labels),
    teacher: parseTeacher(o.teacher),
    budget: parseBudget(o.budget),
    dataset: parseDataset(o.dataset),
    train: parseTrain(o.train),
    eval: parseEval(o.eval),
  };
}

function parseState(raw: unknown): TaskContract["state"] {
  const o = d.object(raw, "$.state", ["fields"]);
  const fields = d.array(o.fields, "$.state.fields", { min: 1, max: 32 }).map((item, i) => {
    const path = `$.state.fields[${i}]`;
    const f = d.object(item, path, ["name", "type", "description", "required", "max_chars"]);

    return {
      name: d.string(f.name, `${path}.name`, { pattern: /^[a-z][a-z0-9_]{0,63}$/ }),
      type: d.oneOf(f.type, `${path}.type`, ["string", "number", "boolean", "json"] as const),
      description: d.string(f.description, `${path}.description`, { max: 1000 }),
      required: f.required === undefined ? true : d.boolean(f.required, `${path}.required`),
      max_chars: d.number(f.max_chars, `${path}.max_chars`, { min: 1, max: 100_000, integer: true }),
    };
  });

  assertUnique(
    fields.map((f) => f.name),
    "$.state.fields[].name",
  );

  return { fields };
}

function parseDecision(raw: unknown): TaskContract["decision"] {
  const o = d.object(raw, "$.decision", ["instructions", "choices"]);
  const instructions = d.entry(o.instructions, "$.decision.instructions");

  const choices = d.array(o.choices, "$.decision.choices", { min: 2, max: 255 }).map((item, i) => {
    const path = `$.decision.choices[${i}]`;
    const c = d.object(item, path, ["label", "description", "criteria"]);
    const description = d.string(c.description, `${path}.description`, { max: 2000 });

    return {
      label: d.string(c.label, `${path}.label`, { pattern: NAME }),
      description,
      criteria: c.criteria === undefined ? description : d.entry(c.criteria, `${path}.criteria`),
    };
  });

  assertUnique(
    choices.map((c) => c.label),
    "$.decision.choices[].label",
  );

  return { instructions, choices };
}

function parseUncertainty(raw: unknown, labels: readonly string[]): UncertaintyPolicy {
  const path = "$.uncertainty";
  const o = d.object(raw, path, ["teacher_min_confidence", "on_low_teacher_confidence", "student_min_confidence", "abstain_label"]);
  const abstain = o.abstain_label === null ? null : d.string(o.abstain_label, `${path}.abstain_label`);

  if (abstain !== null && !labels.includes(abstain)) {
    d.fail(`${path}.abstain_label`, `must be null or one of the decision labels (${labels.join(", ")})`);
  }

  return {
    teacher_min_confidence: d.number(o.teacher_min_confidence, `${path}.teacher_min_confidence`, { min: 0, max: 1 }),
    on_low_teacher_confidence: d.oneOf(o.on_low_teacher_confidence, `${path}.on_low_teacher_confidence`, ["exclude", "keep"] as const),
    student_min_confidence: d.number(o.student_min_confidence, `${path}.student_min_confidence`, { min: 0, max: 1 }),
    abstain_label: abstain,
  };
}

function parseTeacher(raw: unknown): TeacherSpec {
  const path = "$.teacher";
  const o = d.object(raw, path, [
    "model",
    "api_key_env",
    "timeout_ms",
    "max_retries",
    "backoff_initial_ms",
    "backoff_max_ms",
    "concurrency",
    "pricing",
  ]);
  const p = d.object(o.pricing, `${path}.pricing`, ["input_usd_per_mtok", "output_usd_per_mtok", "source", "checked_on"]);
  const initial = d.number(o.backoff_initial_ms, `${path}.backoff_initial_ms`, { min: 0, max: 60_000, integer: true });

  return {
    model: d.string(o.model, `${path}.model`, { pattern: TEACHER_MODEL }),
    api_key_env: d.string(o.api_key_env, `${path}.api_key_env`, { pattern: /^[A-Z_][A-Z0-9_]{0,63}$/ }),
    timeout_ms: d.number(o.timeout_ms, `${path}.timeout_ms`, { min: 100, max: 120_000, integer: true }),
    max_retries: d.number(o.max_retries, `${path}.max_retries`, { min: 0, max: 5, integer: true }),
    backoff_initial_ms: initial,
    backoff_max_ms: d.number(o.backoff_max_ms, `${path}.backoff_max_ms`, { min: initial, max: 60_000, integer: true }),
    concurrency: d.number(o.concurrency, `${path}.concurrency`, { min: 1, max: 8, integer: true }),
    pricing: {
      input_usd_per_mtok: d.number(p.input_usd_per_mtok, `${path}.pricing.input_usd_per_mtok`, { min: 0, max: 1000 }),
      output_usd_per_mtok: d.number(p.output_usd_per_mtok, `${path}.pricing.output_usd_per_mtok`, { min: 0, max: 1000 }),
      source: d.string(p.source, `${path}.pricing.source`, { pattern: /^https:\/\// }),
      checked_on: d.string(p.checked_on, `${path}.pricing.checked_on`, { pattern: /^\d{4}-\d{2}-\d{2}$/ }),
    },
  };
}

function parseBudget(raw: unknown): Budget {
  const path = "$.budget";
  const o = d.object(raw, path, ["max_requests", "max_input_tokens", "max_usd", "max_consecutive_failures"]);

  return {
    max_requests: d.number(o.max_requests, `${path}.max_requests`, { min: 0, max: HARD_LIMITS.maxRequests, integer: true }),
    max_input_tokens: d.number(o.max_input_tokens, `${path}.max_input_tokens`, { min: 0, max: HARD_LIMITS.maxInputTokens, integer: true }),
    max_usd: d.number(o.max_usd, `${path}.max_usd`, { min: 0, max: HARD_LIMITS.maxUsd }),
    max_consecutive_failures: d.number(o.max_consecutive_failures, `${path}.max_consecutive_failures`, { min: 1, max: 100, integer: true }),
  };
}

function parseDataset(raw: unknown): DatasetOptions {
  const path = "$.dataset";
  const o = d.object(raw, path, ["holdout_fraction", "split_seed", "max_rows", "max_line_bytes"]);

  return {
    holdout_fraction: d.number(o.holdout_fraction, `${path}.holdout_fraction`, { min: 0.05, max: 0.9 }),
    split_seed: d.string(o.split_seed, `${path}.split_seed`, { max: 128 }),
    max_rows: d.number(o.max_rows, `${path}.max_rows`, { min: 1, max: 1_000_000, integer: true }),
    max_line_bytes: d.number(o.max_line_bytes, `${path}.max_line_bytes`, { min: 64, max: 1_048_576, integer: true }),
  };
}

function parseTrain(raw: unknown): TrainOptions {
  const path = "$.train";
  const o = d.object(raw, path, ["features", "target", "class_weighting", "l2_grid", "cv_folds", "epochs", "learning_rate"]);
  const grid = d
    .array(o.l2_grid, `${path}.l2_grid`, { min: 1, max: 12 })
    .map((value, i) => d.number(value, `${path}.l2_grid[${i}]`, { min: 1e-9, max: 100 }));

  assertUnique(grid.map(String), `${path}.l2_grid`);

  return {
    features: parseFeatures(o.features),
    target: d.oneOf(o.target, `${path}.target`, ["soft", "hard"] as const),
    class_weighting: d.oneOf(o.class_weighting, `${path}.class_weighting`, ["none", "balanced"] as const),
    l2_grid: [...grid].sort((a, b) => a - b),
    cv_folds: d.number(o.cv_folds, `${path}.cv_folds`, { min: 2, max: 10, integer: true }),
    epochs: d.number(o.epochs, `${path}.epochs`, { min: 1, max: 5000, integer: true }),
    learning_rate: d.number(o.learning_rate, `${path}.learning_rate`, { min: 1e-5, max: 10 }),
  };
}

function parseFeatures(raw: unknown): FeatureOptions {
  const path = "$.train.features";
  const o = d.object(raw, path, ["dimensions", "word_ngrams", "char_ngrams", "per_field"]);
  const dimensions = d.number(o.dimensions, `${path}.dimensions`, { min: 1024, max: 4_194_304, integer: true });

  if ((dimensions & (dimensions - 1)) !== 0) d.fail(`${path}.dimensions`, "must be a power of two");

  let charNgrams: readonly [number, number] | null = null;
  if (o.char_ngrams !== null) {
    const pair = d.array(o.char_ngrams, `${path}.char_ngrams`, { min: 2, max: 2 });
    const lo = d.number(pair[0], `${path}.char_ngrams[0]`, { min: 1, max: 8, integer: true });
    const hi = d.number(pair[1], `${path}.char_ngrams[1]`, { min: lo, max: 8, integer: true });
    charNgrams = [lo, hi];
  }

  return {
    dimensions,
    word_ngrams: d.number(o.word_ngrams, `${path}.word_ngrams`, { min: 1, max: 3, integer: true }),
    char_ngrams: charNgrams,
    per_field: d.boolean(o.per_field, `${path}.per_field`),
  };
}

function parseEval(raw: unknown): EvalOptions {
  const path = "$.eval";
  const o = d.object(raw, path, ["confidence_thresholds", "min_gold_rows"]);
  const thresholds = d
    .array(o.confidence_thresholds, `${path}.confidence_thresholds`, { min: 0, max: 10 })
    .map((value, i) => d.number(value, `${path}.confidence_thresholds[${i}]`, { min: 0, max: 1 }));

  return {
    confidence_thresholds: [...new Set(thresholds)].sort((a, b) => a - b),
    min_gold_rows: d.number(o.min_gold_rows, `${path}.min_gold_rows`, { min: 1, max: 100_000, integer: true }),
  };
}

function assertUnique(values: readonly string[], path: string): void {
  const seen = new Set<string>();

  for (const value of values) {
    if (seen.has(value)) d.fail(path, `duplicate value "${value}"`);
    seen.add(value);
  }
}
