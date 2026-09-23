import { isPinnedModel, type Pricing, type TaskContract } from "./contract.js";
import type { Dataset, DatasetRow } from "./dataset.js";
import { DistillError } from "./errors.js";
import { unitInterval } from "./json.js";
import { appendRecord, currentRecords, type LabelRecord, LEDGER_KIND, LEDGER_VERSION, readLedger, recordKey, spendOf } from "./ledger.js";
import { SDK_ID, type Teacher, type TeacherOutcome } from "./teacher/jev.js";
import { estimateInputTokens, jevQuestion, questionHash } from "./teacher/question.js";

/** Output tokens are free for Jev today; this bound only matters if a contract prices them. */
const WORST_CASE_OUTPUT_TOKENS = 128;

export interface Limits {
  readonly max_requests: number;
  readonly max_input_tokens: number;
  readonly max_usd: number;
}

export interface Money {
  readonly requests: number;
  readonly input_tokens: number;
  readonly usd: number;
}

export type LimitReason = "none" | "max_requests" | "max_input_tokens" | "max_usd";

export interface LabelPlan {
  readonly task: string;
  readonly experimental: boolean;
  readonly teacher_model: string;
  readonly teacher_model_pinned: boolean;
  readonly question_hash: string;
  readonly rows: number;
  readonly already_labeled: number;
  readonly previously_failed: number;
  readonly pending: number;
  readonly selected: number;
  readonly selected_holdout: number;
  readonly limited_by: LimitReason;
  readonly limits: Limits;
  readonly spent_before: Money;
  readonly planned_worst_case: Money;
  /** Rows to send, in deterministic order. Not serialized in reports. */
  readonly queue: readonly { readonly row: DatasetRow; readonly estimate: number }[];
}

export interface PlanOptions {
  readonly retryFailed: boolean;
  /** CLI overrides may only lower the contract budget. */
  readonly overrides: Partial<Limits>;
}

export function usd(inputTokens: number, outputTokens: number, pricing: Pricing): number {
  return (inputTokens * pricing.input_usd_per_mtok + outputTokens * pricing.output_usd_per_mtok) / 1e6;
}

const lower = (base: number, override: number | undefined): number => (override === undefined ? base : Math.min(base, override));

export function effectiveLimits(contract: TaskContract, overrides: Partial<Limits>): Limits {
  return {
    max_requests: lower(contract.budget.max_requests, overrides.max_requests),
    max_input_tokens: lower(contract.budget.max_input_tokens, overrides.max_input_tokens),
    max_usd: lower(contract.budget.max_usd, overrides.max_usd),
  };
}

export function spentMoney(records: readonly LabelRecord[], pricing: Pricing): Money {
  const spend = spendOf(records);
  const output = records.reduce((sum, record) => sum + (record.usage?.output_tokens ?? 0), 0);

  return { requests: spend.requests, input_tokens: spend.input_tokens, usd: usd(spend.input_tokens, output, pricing) };
}

export function planLabeling(contract: TaskContract, dataset: Dataset, ledgerPath: string, options: PlanOptions): LabelPlan {
  const records = readLedger(ledgerPath);
  const hash = questionHash(contract);
  const current = currentRecords(records, hash);
  const question = jevQuestion(contract);
  const limits = effectiveLimits(contract, options.overrides);
  const spent = spentMoney(records, contract.teacher.pricing);

  const statusOf = (row: DatasetRow): LabelRecord["status"] | "none" => current.get(recordKey(row.id, row.contentHash))?.status ?? "none";
  const labeled = dataset.rows.filter((row) => statusOf(row) === "labeled").length;
  const failed = dataset.rows.filter((row) => ["failed", "forbidden"].includes(statusOf(row))).length;
  const pending = dataset.rows
    .filter((row) => statusOf(row) === "none" || (options.retryFailed && statusOf(row) !== "labeled"))
    .sort(
      (a, b) =>
        unitInterval(`${contract.dataset.split_seed}\u0000order\u0000${a.id}`) -
        unitInterval(`${contract.dataset.split_seed}\u0000order\u0000${b.id}`),
    );

  const queue: { row: DatasetRow; estimate: number }[] = [];
  let tokens = 0;
  let limitedBy: LimitReason = "none";

  for (const row of pending) {
    const estimate = estimateInputTokens(row.state, question);
    const nextUsd = spent.usd + usd(tokens + estimate, (queue.length + 1) * WORST_CASE_OUTPUT_TOKENS, contract.teacher.pricing);

    if (spent.requests + queue.length + 1 > limits.max_requests) limitedBy = "max_requests";
    else if (spent.input_tokens + tokens + estimate > limits.max_input_tokens) limitedBy = "max_input_tokens";
    else if (nextUsd > limits.max_usd) limitedBy = "max_usd";
    if (limitedBy !== "none") break;

    queue.push({ row, estimate });
    tokens += estimate;
  }

  return {
    task: contract.name,
    experimental: contract.experimental,
    teacher_model: contract.teacher.model,
    teacher_model_pinned: isPinnedModel(contract.teacher.model),
    question_hash: hash,
    rows: dataset.rows.length,
    already_labeled: labeled,
    previously_failed: failed,
    pending: pending.length,
    selected: queue.length,
    selected_holdout: queue.filter(({ row }) => row.split === "holdout").length,
    limited_by: limitedBy,
    limits,
    spent_before: spent,
    planned_worst_case: {
      requests: queue.length,
      input_tokens: tokens,
      usd: usd(tokens, queue.length * WORST_CASE_OUTPUT_TOKENS, contract.teacher.pricing),
    },
    queue,
  };
}

export type StopReason = "completed" | "consecutive_failures" | "fatal" | "budget";

export interface LabelRunSummary {
  readonly attempted: number;
  readonly labeled: number;
  readonly forbidden: number;
  readonly failed: number;
  readonly stopped: StopReason;
  readonly fatal_message: string | null;
  readonly http_attempts: number;
  readonly spent_this_run: Money;
  readonly answered_models: readonly string[];
}

export interface RunOptions {
  readonly approve: boolean;
  readonly teacher: Teacher;
  readonly now?: () => string;
}

export async function runLabeling(
  contract: TaskContract,
  plan: LabelPlan,
  ledgerPath: string,
  options: RunOptions,
): Promise<LabelRunSummary> {
  if (!options.approve)
    throw new DistillError("approval_required", "labeling sends paid Jev requests; re-run with --approve after reviewing the plan");

  const now = options.now ?? (() => new Date().toISOString());
  const queue = [...plan.queue];
  const counts = { labeled: 0, forbidden: 0, failed: 0, consecutive: 0, inputTokens: 0, outputTokens: 0, requests: 0 };
  const models = new Set<string>();
  let stopped: StopReason = "completed";
  let fatal: string | null = null;

  const worker = async (): Promise<void> => {
    for (let item = queue.shift(); item !== undefined && stopped === "completed"; item = queue.shift()) {
      if (!withinBudget(plan, counts, item.estimate, contract.teacher.pricing)) {
        stopped = "budget";
        return;
      }

      counts.requests += 1;
      counts.inputTokens += item.estimate;
      const { outcome, attempts } = await options.teacher.ask(item.row.state);

      if (outcome.kind === "fatal") {
        counts.requests -= 1;
        counts.inputTokens -= item.estimate;
        stopped = "fatal";
        fatal = outcome.message;
        return;
      }

      const record = toRecord(contract, plan, item.row, item.estimate, outcome, attempts, now());
      counts.inputTokens += record.charged_input_tokens - item.estimate;
      counts.outputTokens += record.usage?.output_tokens ?? 0;
      appendRecord(ledgerPath, record);
      tally(counts, record, models);

      if (counts.consecutive >= contract.budget.max_consecutive_failures) stopped = "consecutive_failures";
    }
  };

  await Promise.all(Array.from({ length: contract.teacher.concurrency }, worker));

  return {
    attempted: counts.labeled + counts.forbidden + counts.failed,
    labeled: counts.labeled,
    forbidden: counts.forbidden,
    failed: counts.failed,
    stopped,
    fatal_message: fatal,
    http_attempts: options.teacher.attempts(),
    spent_this_run: {
      requests: counts.requests,
      input_tokens: counts.inputTokens,
      usd: usd(counts.inputTokens, counts.outputTokens, contract.teacher.pricing),
    },
    answered_models: [...models].sort(),
  };
}

interface Counts {
  labeled: number;
  forbidden: number;
  failed: number;
  consecutive: number;
  inputTokens: number;
  outputTokens: number;
  requests: number;
}

function withinBudget(plan: LabelPlan, counts: Counts, estimate: number, pricing: Pricing): boolean {
  const { limits, spent_before: spent } = plan;
  const inputTokens = spent.input_tokens + counts.inputTokens + estimate;
  const worstUsd = spent.usd + usd(counts.inputTokens + estimate, counts.outputTokens + WORST_CASE_OUTPUT_TOKENS, pricing);

  return (
    spent.requests + counts.requests + 1 <= limits.max_requests && inputTokens <= limits.max_input_tokens && worstUsd <= limits.max_usd
  );
}

function tally(counts: Counts, record: LabelRecord, models: Set<string>): void {
  if (record.answered_model !== null) models.add(record.answered_model);

  if (record.status === "labeled") {
    counts.labeled += 1;
    counts.consecutive = 0;
    return;
  }

  counts.consecutive += 1;
  if (record.status === "forbidden") counts.forbidden += 1;
  else counts.failed += 1;
}

type AskedOutcome = Exclude<TeacherOutcome, { kind: "fatal" }>;

function toRecord(
  contract: TaskContract,
  plan: LabelPlan,
  row: DatasetRow,
  estimate: number,
  outcome: AskedOutcome,
  attempts: number,
  at: string,
): LabelRecord {
  const base = {
    ledger: LEDGER_KIND,
    ledger_version: LEDGER_VERSION,
    id: row.id,
    content_hash: row.contentHash,
    question_hash: plan.question_hash,
    requested_model: contract.teacher.model,
    attempts,
    sdk: SDK_ID,
    labeled_at: at,
  } as const;

  if (outcome.kind === "answered") {
    return {
      ...base,
      status: "labeled",
      choice: outcome.choice,
      probabilities: outcome.probabilities,
      confidence: outcome.confidence,
      answered_model: outcome.answeredModel,
      usage: outcome.usage,
      charged_input_tokens: outcome.usage.input_tokens,
      request_id: outcome.requestId,
      error: null,
    };
  }

  // No label is ever synthesized for a failure; the row is simply unlabeled.
  return {
    ...base,
    status: outcome.kind === "forbidden" ? "forbidden" : "failed",
    choice: null,
    probabilities: null,
    confidence: null,
    answered_model: null,
    usage: null,
    charged_input_tokens: estimate,
    request_id: outcome.requestId,
    error:
      outcome.kind === "forbidden"
        ? { reason: "forbidden", status: 403, message: outcome.message }
        : { reason: outcome.reason, status: outcome.status, message: outcome.message },
  };
}
