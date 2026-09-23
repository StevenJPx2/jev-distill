import type { StateField } from "./contract.js";
import { DistillError } from "./errors.js";
import { canonicalJson, isJsonValue, isPlainObject, type JsonObject, type JsonValue, sha256Hex } from "./json.js";

/** A state object that passed the contract's field schema; keys follow schema order. */
export type TaskState = JsonObject;

export function validateState(fields: readonly StateField[], raw: unknown, where: string): TaskState {
  if (!isPlainObject(raw)) throw new DistillError("invalid_state", `${where}: state must be an object`);

  const declared = new Set(fields.map((field) => field.name));
  const unknown = Object.keys(raw).filter((key) => !declared.has(key));
  if (unknown.length > 0) throw new DistillError("invalid_state", `${where}: undeclared state field(s): ${unknown.join(", ")}`);

  const state: TaskState = {};

  for (const field of fields) {
    const value = raw[field.name];

    if (value === undefined || value === null) {
      if (field.required) throw new DistillError("invalid_state", `${where}: missing required field "${field.name}"`);
      continue;
    }

    state[field.name] = checkField(field, value, where);
  }

  return state;
}

function checkField(field: StateField, value: unknown, where: string): JsonValue {
  const fail = (message: string): never => {
    throw new DistillError("invalid_state", `${where}: field "${field.name}" ${message}`);
  };

  const matches =
    (field.type === "string" && typeof value === "string") ||
    (field.type === "number" && typeof value === "number" && Number.isFinite(value)) ||
    (field.type === "boolean" && typeof value === "boolean") ||
    (field.type === "json" && isJsonValue(value));

  if (!matches) fail(`must be of type ${field.type}`);

  const json = value as JsonValue;
  const size = typeof json === "string" ? json.length : canonicalJson(json).length;
  if (size > field.max_chars) fail(`exceeds max_chars (${size} > ${field.max_chars})`);

  return json;
}

/** Text form of one field value, shared by hashing and featurization. */
export function fieldText(value: JsonValue): string {
  return typeof value === "string" ? value : canonicalJson(value);
}

export function normalizeText(text: string): string {
  return text.normalize("NFKC").toLowerCase().replace(/\s+/gu, " ").trim();
}

/**
 * Identity of the state's content for dedupe, split assignment, and label
 * staleness. Case and whitespace differences are treated as the same content.
 */
export function contentHash(state: TaskState): string {
  const normalized: JsonObject = {};
  for (const [key, value] of Object.entries(state)) normalized[key] = normalizeText(fieldText(value));

  return sha256Hex(canonicalJson(normalized));
}
