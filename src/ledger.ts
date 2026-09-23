import { appendFileSync, existsSync } from "node:fs";

import { readJsonLines } from "./dataset.js";
import { Decoder } from "./json.js";
import type { FailureReason, Usage } from "./teacher/jev.js";

export const LEDGER_KIND = "jev-distill.labels";
export const LEDGER_VERSION = 1;

export type LabelStatus = "labeled" | "forbidden" | "failed";

/**
 * One Jev call for one row. Stores hashes, the teacher's answer, and
 * accounting; never the state text. Failed and forbidden rows have no label.
 */
export interface LabelRecord {
  readonly ledger: typeof LEDGER_KIND;
  readonly ledger_version: typeof LEDGER_VERSION;
  readonly id: string;
  readonly content_hash: string;
  readonly question_hash: string;
  readonly status: LabelStatus;
  readonly choice: string | null;
  readonly probabilities: Readonly<Record<string, number>> | null;
  readonly confidence: number | null;
  readonly requested_model: string;
  readonly answered_model: string | null;
  readonly usage: Usage | null;
  /** Tokens counted against the budget: reported usage, or the worst-case estimate when none was reported. */
  readonly charged_input_tokens: number;
  readonly request_id: string | null;
  readonly error: { readonly reason: FailureReason | "forbidden"; readonly status: number | null; readonly message: string } | null;
  readonly attempts: number;
  readonly sdk: string;
  readonly labeled_at: string;
}

export interface Spend {
  readonly requests: number;
  readonly input_tokens: number;
}

const d = new Decoder("invalid_ledger");
const FIELDS = [
  "ledger",
  "ledger_version",
  "id",
  "content_hash",
  "question_hash",
  "status",
  "choice",
  "probabilities",
  "confidence",
  "requested_model",
  "answered_model",
  "usage",
  "charged_input_tokens",
  "request_id",
  "error",
  "attempts",
  "sdk",
  "labeled_at",
] as const;

export function readLedger(path: string): LabelRecord[] {
  if (!existsSync(path)) return [];

  return readJsonLines(path, 65_536, 10_000_000).map(({ line, value }) => decodeRecord(value, `${path}:${line}`));
}

export function appendRecord(path: string, record: LabelRecord): void {
  appendFileSync(path, `${JSON.stringify(record)}\n`, "utf8");
}

export function spendOf(records: readonly LabelRecord[]): Spend {
  return {
    requests: records.length,
    input_tokens: records.reduce((sum, record) => sum + record.charged_input_tokens, 0),
  };
}

/** Latest record per row id that still matches the row's content and the current question. */
export function currentRecords(records: readonly LabelRecord[], questionHash: string): Map<string, LabelRecord> {
  const latest = new Map<string, LabelRecord>();
  for (const record of records) if (record.question_hash === questionHash) latest.set(recordKey(record.id, record.content_hash), record);

  return latest;
}

export function recordKey(id: string, contentHash: string): string {
  return `${id}\u0000${contentHash}`;
}

function decodeRecord(value: unknown, where: string): LabelRecord {
  const o = d.object(value, where, FIELDS);
  if (o.ledger !== LEDGER_KIND || o.ledger_version !== LEDGER_VERSION) d.fail(where, `not a ${LEDGER_KIND} v${LEDGER_VERSION} record`);

  const status = d.oneOf(o.status, `${where}.status`, ["labeled", "forbidden", "failed"] as const);
  const labeled = status === "labeled";

  if (labeled !== (typeof o.choice === "string")) d.fail(`${where}.choice`, "a choice is present exactly when status is labeled");
  if (labeled && (o.probabilities === null || typeof o.confidence !== "number"))
    d.fail(where, "labeled records need probabilities and confidence");

  d.number(o.charged_input_tokens, `${where}.charged_input_tokens`, { min: 0, max: 1e9, integer: true });
  d.number(o.attempts, `${where}.attempts`, { min: 0, max: 100, integer: true });
  d.string(o.id, `${where}.id`);
  d.string(o.content_hash, `${where}.content_hash`, { pattern: /^[0-9a-f]{64}$/ });
  d.string(o.question_hash, `${where}.question_hash`, { pattern: /^[0-9a-f]{64}$/ });

  // Safety: every field was checked above or is informational text written by appendRecord.
  return o as unknown as LabelRecord;
}
