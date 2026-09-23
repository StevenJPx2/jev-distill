import { createHash } from "node:crypto";

import { DistillError, type ErrorCode } from "./errors.js";

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
export type JsonObject = { [key: string]: JsonValue };
/** What Jev accepts for instructions and criteria: text, an object, or an array. */
export type Entry = string | JsonObject | JsonValue[];

export function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function isJsonValue(value: unknown, depth = 0): value is JsonValue {
  if (depth > 32) return false;
  if (value === null || typeof value === "string" || typeof value === "boolean") return true;
  if (typeof value === "number") return Number.isFinite(value);
  if (Array.isArray(value)) return value.every((item) => isJsonValue(item, depth + 1));
  if (isPlainObject(value)) return Object.values(value).every((item) => isJsonValue(item, depth + 1));

  return false;
}

/** JSON with object keys sorted recursively, so equal values hash equally on every machine. */
export function canonicalJson(value: JsonValue): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const keys = Object.keys(value).sort();
    const entries = keys.map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key] as JsonValue)}`);

    return `{${entries.join(",")}}`;
  }

  return JSON.stringify(value);
}

export function sha256Hex(text: string | Uint8Array): string {
  return createHash("sha256").update(text).digest("hex");
}

/** Deterministic value in [0, 1) derived from a string; used for splits and ordering. */
export function unitInterval(text: string): number {
  const hex = sha256Hex(text).slice(0, 13);

  return Number.parseInt(hex, 16) / 2 ** 52;
}

export function parseJson(text: string, code: ErrorCode, where: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch (error) {
    throw new DistillError(code, `${where}: invalid JSON (${(error as Error).message})`);
  }
}

/**
 * Strict decoding helper: every object rejects keys it does not declare, so a
 * typo fails loudly instead of silently falling back to a default.
 */
export class Decoder {
  readonly code: ErrorCode;

  constructor(code: ErrorCode) {
    this.code = code;
  }

  fail(path: string, message: string): never {
    throw new DistillError(this.code, `${path}: ${message}`);
  }

  object(value: unknown, path: string, allowed: readonly string[]): Record<string, unknown> {
    if (!isPlainObject(value)) this.fail(path, "expected an object");

    const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
    if (unknown.length > 0) this.fail(path, `unknown field(s): ${unknown.join(", ")}`);

    return value;
  }

  string(value: unknown, path: string, options: { min?: number; max?: number; pattern?: RegExp } = {}): string {
    if (typeof value !== "string") this.fail(path, "expected a string");
    if (value.length < (options.min ?? 1)) this.fail(path, `must have at least ${options.min ?? 1} character(s)`);
    if (options.max !== undefined && value.length > options.max) this.fail(path, `must have at most ${options.max} characters`);
    if (options.pattern && !options.pattern.test(value)) this.fail(path, `must match ${String(options.pattern)}`);

    return value;
  }

  number(value: unknown, path: string, range: { min: number; max: number; integer?: boolean }): number {
    if (typeof value !== "number" || !Number.isFinite(value)) this.fail(path, "expected a finite number");
    if (range.integer === true && !Number.isInteger(value)) this.fail(path, "expected an integer");
    if (value < range.min || value > range.max) this.fail(path, `must be between ${range.min} and ${range.max}`);

    return value;
  }

  boolean(value: unknown, path: string): boolean {
    if (typeof value !== "boolean") this.fail(path, "expected a boolean");

    return value;
  }

  oneOf<T extends string>(value: unknown, path: string, options: readonly T[]): T {
    const found = options.find((option) => option === value);
    if (found === undefined) this.fail(path, `must be one of: ${options.join(", ")}`);

    return found;
  }

  array(value: unknown, path: string, bounds: { min: number; max: number }): unknown[] {
    if (!Array.isArray(value)) this.fail(path, "expected an array");
    if (value.length < bounds.min || value.length > bounds.max) {
      this.fail(path, `must have between ${bounds.min} and ${bounds.max} entries`);
    }

    return value;
  }

  json(value: unknown, path: string): JsonValue {
    if (!isJsonValue(value)) this.fail(path, "expected JSON data (finite numbers, depth <= 32)");

    return value;
  }

  entry(value: unknown, path: string): Entry {
    const json = this.json(value, path);
    if (json === "" || json === null || typeof json === "number" || typeof json === "boolean") {
      this.fail(path, "expected non-empty text, an object, or an array");
    }

    return json;
  }
}
