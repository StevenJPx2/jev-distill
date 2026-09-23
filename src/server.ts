import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";

import { DistillError } from "./errors.js";
import { isPlainObject } from "./json.js";
import type { Predictor } from "./student/index.js";

const MAX_BODY_BYTES = 256 * 1024;

/**
 * Local HTTP front for student artifacts, for callers that are not Node.
 * Pure inference: no Jev client, no key, no outbound network.
 */
export function createStudentServer(predictors: ReadonlyMap<string, Predictor>): Server {
  return createServer((req, res) => {
    handle(predictors, req, res).catch((error: unknown) => send(res, 500, { error: "internal", message: (error as Error).message }));
  });
}

async function handle(predictors: ReadonlyMap<string, Predictor>, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? "/", "http://localhost");

  if (req.method === "GET" && url.pathname === "/healthz") {
    const students = [...predictors.values()].map(({ artifact: { task } }) => ({
      name: task.name,
      version: task.version,
      experimental: task.experimental,
      labels: task.labels,
    }));
    send(res, 200, { status: "ok", students });
    return;
  }

  const match = /^\/v1\/predict\/([a-z][a-z0-9_-]{0,63})$/.exec(url.pathname);
  if (req.method !== "POST" || !match) {
    send(res, 404, { error: "not_found", message: "use GET /healthz or POST /v1/predict/<task>" });
    return;
  }

  const predictor = predictors.get(match[1] ?? "");
  if (!predictor) {
    send(res, 404, { error: "unknown_task", message: `no student named ${match[1] ?? ""}` });
    return;
  }

  const body = await readBody(req);
  if (body === null) {
    send(res, 413, { error: "too_large", message: `body exceeds ${MAX_BODY_BYTES} bytes` });
    return;
  }

  try {
    const parsed: unknown = JSON.parse(body);
    if (!isPlainObject(parsed) || Object.keys(parsed).some((key) => key !== "state"))
      throw new DistillError("invalid_state", 'body must be {"state": {...}}');

    send(res, 200, predictor.predict(parsed.state));
  } catch (error) {
    send(res, 400, { error: error instanceof DistillError ? error.code : "invalid_json", message: (error as Error).message });
  }
}

async function readBody(req: IncomingMessage): Promise<string | null> {
  const chunks: Buffer[] = [];
  let size = 0;

  // Keep draining past the limit (without buffering) so the 413 reply is delivered.
  for await (const chunk of req) {
    const buffer = chunk as Buffer;
    size += buffer.length;
    if (size <= MAX_BODY_BYTES) chunks.push(buffer);
  }

  return size > MAX_BODY_BYTES ? null : Buffer.concat(chunks).toString("utf8");
}

function send(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(`${JSON.stringify(body)}\n`);
}
