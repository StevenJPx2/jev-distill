import type { Fetch } from "@typesafe-ai/sdk";

/**
 * Offline stand-in for the TypeSafe HTTP API. It is plugged into the real SDK
 * through its `fetch` option, so request building, retries, and error classes
 * are the SDK's own. Behavior is decided per request from the state text.
 */
export interface FakeJevOptions {
  /** Label and unnormalized scores for a state; return null to answer uniformly. */
  readonly classify: (text: string, labels: readonly string[]) => Record<string, number>;
  /** Content-specific 403 (the observed behavior this tool must survive). */
  readonly forbid?: (text: string) => boolean;
  /** Transient failures before success, keyed by text match. */
  readonly flaky?: (text: string, attempt: number) => number | null;
  /** Force a malformed answer. */
  readonly malformed?: (text: string) => boolean;
  /** Every request fails with this status (e.g. 401). */
  readonly always?: number;
  readonly model?: string;
}

export interface FakeJev {
  readonly fetch: Fetch;
  readonly requests: { readonly url: string; readonly body: Record<string, unknown>; readonly authorization: string | null }[];
}

const json = (status: number, body: unknown, extra: Record<string, string> = {}): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json",
      "x-typesafe-request-id": `req_${status}_${Math.random().toString(36).slice(2, 8)}`,
      ...extra,
    },
  });

export function fakeJev(options: FakeJevOptions): FakeJev {
  const requests: FakeJev["requests"][number][] = [];
  const attemptsByText = new Map<string, number>();

  const fetch: Fetch = async (url, init) => {
    const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
    const headers = new Headers(init?.headers);
    requests.push({ url, body, authorization: headers.get("authorization") });

    if (options.always !== undefined) return json(options.always, { error: { message: `status ${options.always}` } });

    const text = JSON.stringify(body.state);
    const attempt = attemptsByText.get(text) ?? 0;
    attemptsByText.set(text, attempt + 1);

    const transient = options.flaky?.(text, attempt) ?? null;
    if (transient !== null) return json(transient, { error: { message: "try again" } }, { "retry-after-ms": "1" });
    if (options.forbid?.(text)) return json(403, { detail: "Request blocked by content policy." });

    const question = (body.questions as Record<string, { criteria: Record<string, unknown> }>).decision;
    const labels = Object.keys(question?.criteria ?? {});
    const scores = options.classify(text.toLowerCase(), labels);
    const probs = normalize(labels, scores);
    const choice = labels.reduce((best, label) => ((probs[label] ?? 0) > (probs[best] ?? 0) ? label : best), labels[0] ?? "");
    const peak = probs[choice] ?? 0;

    return json(200, {
      model: options.model ?? "jev-1.13.0",
      answers: {
        decision: {
          type: "choice",
          choice: options.malformed?.(text) ? "not-a-label" : choice,
          probabilities: probs,
          confidence: Math.max(0, (labels.length * peak - 1) / (labels.length - 1)),
        },
      },
      usage: { input_tokens: 300 + Math.ceil(text.length / 4), output_tokens: 20 },
    });
  };

  return { fetch, requests };
}

function normalize(labels: readonly string[], scores: Record<string, number>): Record<string, number> {
  const exps = labels.map((label) => Math.exp(scores[label] ?? 0));
  const total = exps.reduce((sum, value) => sum + value, 0);

  return Object.fromEntries(labels.map((label, i) => [label, Math.round(((exps[i] ?? 0) / total) * 1e6) / 1e6]));
}

const KEYWORDS: Record<string, readonly string[]> = {
  billing: [
    "charge",
    "refund",
    "invoice",
    "billed",
    "billing",
    "payment",
    "card",
    "receipt",
    "tax",
    "currency",
    "prorat",
    "coupon",
    "renewal",
  ],
  technical: ["error", "crash", "api", "bug", "fail", "broken", "sync", "webhook", "500", "502", "slow", "integration", "stopped", "404"],
  account: ["account", "login", "password", "2fa", "delete", "username", "profile", "workspace", "admin", "gdpr", "access", "role"],
  sales: ["pricing", "price", "demo", "discount", "quote", "trial", "contract", "enterprise", "seats", "buy", "purchas", "licens"],
};

/** Keyword teacher: deliberately imperfect, so teacher agreement and gold accuracy differ. */
export function routingTeacher(text: string, labels: readonly string[]): Record<string, number> {
  return Object.fromEntries(labels.map((label) => [label, 1.2 * (KEYWORDS[label] ?? []).filter((word) => text.includes(word)).length]));
}
