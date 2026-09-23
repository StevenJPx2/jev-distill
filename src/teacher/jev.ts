import {
  APIConnectionError,
  APIError,
  APITimeoutError,
  type Fetch,
  TypeSafeClient,
  TypeSafeError,
  VERSION as SDK_VERSION,
} from "@typesafe-ai/sdk";

import { labelsOf, type TaskContract } from "../contract.js";
import { isPlainObject } from "../json.js";
import type { TaskState } from "../state.js";
import { jevQuestion, QUESTION_KEY } from "./question.js";

export const SDK_ID = `@typesafe-ai/sdk@${SDK_VERSION}`;

export interface Usage {
  readonly input_tokens: number;
  readonly output_tokens: number;
}

export type FailureReason =
  | "rate_limited"
  | "server_error"
  | "timeout"
  | "connection"
  | "bad_request"
  | "not_found"
  | "http_error"
  | "invalid_response";

/** Result of asking Jev about one row. Only `answered` may ever become a label. */
export type TeacherOutcome =
  | {
      readonly kind: "answered";
      readonly choice: string;
      readonly probabilities: Readonly<Record<string, number>>;
      readonly confidence: number;
      readonly answeredModel: string;
      readonly usage: Usage;
      readonly requestId: string | null;
    }
  /** HTTP 403 for this content. Observed to be content-specific, so the row fails and the run continues. */
  | { readonly kind: "forbidden"; readonly requestId: string | null; readonly message: string }
  | {
      readonly kind: "failed";
      readonly reason: FailureReason;
      readonly status: number | null;
      readonly requestId: string | null;
      readonly message: string;
    }
  /** Account-level problem (e.g. 401); no further request in this run can succeed. */
  | { readonly kind: "fatal"; readonly status: number | null; readonly message: string };

export interface Asked {
  readonly outcome: TeacherOutcome;
  /** HTTP attempts for this call, including SDK retries. */
  readonly attempts: number;
}

export interface Teacher {
  ask(state: TaskState, signal?: AbortSignal): Promise<Asked>;
  /** HTTP attempts made so far across all calls. */
  attempts(): number;
}

export interface JevTeacherOptions {
  readonly apiKey: string;
  /** Transport seam for offline tests; defaults to global fetch. */
  readonly fetch?: Fetch;
  readonly baseURL?: string;
}

const RETRY_STATUSES = [408, 429, ...Array.from({ length: 100 }, (_, i) => 500 + i)];

export function createJevTeacher(contract: TaskContract, options: JevTeacherOptions): Teacher {
  let total = 0;
  const transport: Fetch = options.fetch ?? ((input, init) => globalThis.fetch(input, init));
  const question = jevQuestion(contract);
  const labels = labelsOf(contract);

  // One lightweight client per call, so retries are attributed to the right row under concurrency.
  const clientFor = (counter: { attempts: number }): TypeSafeClient =>
    new TypeSafeClient({
      apiKey: options.apiKey,
      ...(options.baseURL === undefined ? {} : { baseURL: options.baseURL }),
      defaultModel: contract.teacher.model,
      timeout: contract.teacher.timeout_ms,
      // Bodies contain task state; never let the SDK log them.
      logLevel: "off",
      retry: {
        maxRetries: contract.teacher.max_retries,
        backoffInitialMs: contract.teacher.backoff_initial_ms,
        backoffMaxMs: contract.teacher.backoff_max_ms,
        maxRetryAfterMs: contract.teacher.backoff_max_ms,
        httpStatuses: new Set(RETRY_STATUSES),
      },
      fetch: (input, init) => {
        counter.attempts += 1;
        total += 1;
        return transport(input, init);
      },
    });

  return {
    attempts: () => total,
    async ask(state, signal) {
      const counter = { attempts: 0 };
      try {
        const { data, requestId } = await clientFor(counter)
          .systemOne({ model: contract.teacher.model, state, questions: { [QUESTION_KEY]: question } }, signal ? { signal } : {})
          .withResponse();

        return { outcome: decodeAnswer(data, labels, requestId ?? null), attempts: counter.attempts };
      } catch (error) {
        return { outcome: classifyError(error), attempts: counter.attempts };
      }
    },
  };
}

function decodeAnswer(data: unknown, labels: readonly string[], requestId: string | null): TeacherOutcome {
  const invalid = (message: string): TeacherOutcome => ({ kind: "failed", reason: "invalid_response", status: null, requestId, message });

  if (!isPlainObject(data) || !isPlainObject(data.answers) || !isPlainObject(data.usage)) return invalid("missing answers or usage");

  const answer = data.answers[QUESTION_KEY];
  if (!isPlainObject(answer) || answer.type !== "choice") return invalid("missing choice answer");
  if (typeof answer.choice !== "string" || !labels.includes(answer.choice)) return invalid("choice is not a declared label");
  if (typeof answer.confidence !== "number" || !(answer.confidence >= 0 && answer.confidence <= 1))
    return invalid("confidence out of range");
  if (!isPlainObject(answer.probabilities)) return invalid("missing probabilities");

  const probs = answer.probabilities;
  const keys = Object.keys(probs);
  if (keys.length !== labels.length || !labels.every((label) => typeof probs[label] === "number")) {
    return invalid("probabilities do not cover exactly the declared labels");
  }

  const values = labels.map((label) => probs[label] as number);
  const total = values.reduce((sum, value) => sum + value, 0);
  if (!values.every((value) => value >= 0 && value <= 1) || Math.abs(total - 1) > 0.02)
    return invalid("probabilities are not a distribution");

  const { input_tokens: input, output_tokens: output } = data.usage;
  if (typeof input !== "number" || typeof output !== "number") return invalid("usage is missing token counts");
  if (typeof data.model !== "string" || data.model === "") return invalid("missing answering model");

  return {
    kind: "answered",
    choice: answer.choice,
    probabilities: Object.fromEntries(labels.map((label, i) => [label, (values[i] ?? 0) / total])),
    confidence: answer.confidence,
    answeredModel: data.model,
    usage: { input_tokens: input, output_tokens: output },
    requestId,
  };
}

function classifyError(error: unknown): TeacherOutcome {
  if (error instanceof APIError) {
    // Edge rejections (observed: Cloudflare HTML 403s for some request content) carry cf-ray, not a TypeSafe request ID.
    const ray = error.headers.get("cf-ray");
    const requestId = error.requestId ?? (ray === null ? null : `cf-ray:${ray}`);
    const html = typeof error.body === "string" && /^\s*<(!doctype|html)/i.test(error.body);
    const message = html
      ? `${error.status} HTML page from ${error.headers.get("server") ?? "edge"} (request blocked before reaching Jev)`
      : redact(error.message);

    if (error.status === 401) return { kind: "fatal", status: 401, message };
    if (error.status === 403) return { kind: "forbidden", requestId, message };

    return { kind: "failed", reason: reasonForStatus(error.status), status: error.status, requestId, message };
  }

  if (error instanceof APITimeoutError)
    return { kind: "failed", reason: "timeout", status: null, requestId: null, message: redact(error.message) };
  if (error instanceof APIConnectionError)
    return { kind: "failed", reason: "connection", status: null, requestId: null, message: redact(error.message) };
  if (error instanceof TypeSafeError) return { kind: "fatal", status: null, message: redact(error.message) };

  throw error;
}

function reasonForStatus(status: number): FailureReason {
  if (status === 429) return "rate_limited";
  if (status === 408) return "timeout";
  if (status >= 500) return "server_error";
  if (status === 404) return "not_found";
  if (status === 400 || status === 422) return "bad_request";

  return "http_error";
}

/** Error text may echo request content; keep it short and strip obvious identifiers. */
export function redact(message: string): string {
  return message
    .replace(/[\w.+-]+@[\w-]+\.[\w.-]+/g, "[email]")
    .replace(/[A-Za-z0-9_-]{24,}/g, "[token]")
    .slice(0, 160);
}
