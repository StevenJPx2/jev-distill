import { readFileSync } from "node:fs";

import type { StateField, UncertaintyPolicy } from "../contract.js";
import { DistillError } from "../errors.js";
import { canonicalJson, Decoder, isPlainObject, type JsonObject, parseJson, sha256Hex } from "../json.js";
import type { FeatureSpec } from "./features.js";
import type { LinearModel } from "./model.js";

export const ARTIFACT_FORMAT = "jev-distill.student";
export const ARTIFACT_VERSION = 1;

export interface ArtifactTask {
  readonly name: string;
  readonly version: string;
  readonly contract_hash: string;
  readonly experimental: boolean;
  readonly labels: readonly string[];
  readonly state_fields: readonly StateField[];
  readonly uncertainty: Pick<UncertaintyPolicy, "student_min_confidence" | "abstain_label">;
}

/** A self-contained student: no Jev, Laya, network, or original data needed to predict. */
export interface StudentArtifact {
  readonly format: typeof ARTIFACT_FORMAT;
  readonly format_version: typeof ARTIFACT_VERSION;
  readonly task: ArtifactTask;
  readonly features: FeatureSpec;
  readonly model: LinearModel;
  /** Training provenance and CV record; informational, covered by the integrity hash. */
  readonly provenance: JsonObject;
}

function encodeInt32(values: Int32Array): string {
  const view = new DataView(new ArrayBuffer(values.length * 4));
  values.forEach((value, i) => view.setInt32(i * 4, value, true));

  return Buffer.from(view.buffer).toString("base64");
}

function encodeFloat64(values: Float64Array): string {
  const view = new DataView(new ArrayBuffer(values.length * 8));
  values.forEach((value, i) => view.setFloat64(i * 8, value, true));

  return Buffer.from(view.buffer).toString("base64");
}

function decodeBytes(text: unknown, width: number, path: string): DataView {
  if (typeof text !== "string") throw new DistillError("invalid_artifact", `${path}: expected base64`);

  const bytes = Buffer.from(text, "base64");
  if (bytes.length % width !== 0) throw new DistillError("invalid_artifact", `${path}: truncated`);

  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

/** JSON form with explicit little-endian binary blocks, identical on every platform. */
export function artifactToJson(artifact: StudentArtifact): JsonObject {
  const body: JsonObject = {
    format: artifact.format,
    format_version: artifact.format_version,
    task: artifact.task as unknown as JsonObject,
    features: artifact.features as unknown as JsonObject,
    model: {
      encoding: "base64-little-endian",
      labels: [...artifact.model.labels],
      columns_int32: encodeInt32(artifact.model.columns),
      weights_float64: encodeFloat64(artifact.model.weights),
      bias: [...artifact.model.bias],
    },
    provenance: artifact.provenance,
  };

  return { ...body, integrity: { algorithm: "sha256-canonical-json", digest: sha256Hex(canonicalJson(body)) } };
}

export function loadArtifact(path: string): StudentArtifact {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    throw new DistillError("io", `cannot read artifact ${path}: ${(error as Error).message}`);
  }

  return parseArtifact(parseJson(text, "invalid_artifact", path));
}

export function parseArtifact(raw: unknown): StudentArtifact {
  const d = new Decoder("invalid_artifact");
  const o = d.object(raw, "$", ["format", "format_version", "task", "features", "model", "provenance", "integrity"]);

  if (o.format !== ARTIFACT_FORMAT) d.fail("$.format", `must be "${ARTIFACT_FORMAT}"`);
  if (o.format_version !== ARTIFACT_VERSION) d.fail("$.format_version", `unsupported; this build reads version ${ARTIFACT_VERSION}`);

  const { integrity, ...body } = o;
  const seal = d.object(integrity, "$.integrity", ["algorithm", "digest"]);
  if (seal.digest !== sha256Hex(canonicalJson(d.json(body, "$"))))
    d.fail("$.integrity", "digest mismatch; artifact was modified or corrupted");

  const task = d.object(o.task, "$.task", ["name", "version", "contract_hash", "experimental", "labels", "state_fields", "uncertainty"]);
  const model = decodeModel(d, o.model);
  const labels = d.array(task.labels, "$.task.labels", { min: 2, max: 255 }).map((label, i) => d.string(label, `$.task.labels[${i}]`));

  if (canonicalJson(labels) !== canonicalJson([...model.labels])) d.fail("$.model.labels", "must equal $.task.labels");
  if (!isPlainObject(o.features) || o.features.kind !== "hashed-ngrams" || o.features.version !== 1) {
    d.fail("$.features", "unsupported feature kind or version");
  }

  // Safety: the integrity digest proves these blocks were written by artifactToJson from typed values.
  return {
    format: ARTIFACT_FORMAT,
    format_version: ARTIFACT_VERSION,
    task: task as unknown as ArtifactTask,
    features: o.features as unknown as FeatureSpec,
    model,
    provenance: d.json(o.provenance, "$.provenance") as JsonObject,
  };
}

function decodeModel(d: Decoder, raw: unknown): LinearModel {
  const m = d.object(raw, "$.model", ["encoding", "labels", "columns_int32", "weights_float64", "bias"]);
  if (m.encoding !== "base64-little-endian") d.fail("$.model.encoding", "unsupported");

  const labels = d.array(m.labels, "$.model.labels", { min: 2, max: 255 }).map((label, i) => d.string(label, `$.model.labels[${i}]`));
  const columnsView = decodeBytes(m.columns_int32, 4, "$.model.columns_int32");
  const weightsView = decodeBytes(m.weights_float64, 8, "$.model.weights_float64");
  const columns = Int32Array.from({ length: columnsView.byteLength / 4 }, (_, i) => columnsView.getInt32(i * 4, true));
  const weights = Float64Array.from({ length: weightsView.byteLength / 8 }, (_, i) => weightsView.getFloat64(i * 8, true));
  const bias = Float64Array.from(d.array(m.bias, "$.model.bias", { min: labels.length, max: labels.length }), (value, i) =>
    d.number(value, `$.model.bias[${i}]`, { min: -1e12, max: 1e12 }),
  );

  if (weights.length !== columns.length * labels.length) d.fail("$.model", "weights do not match columns x labels");
  if (!weights.every(Number.isFinite)) d.fail("$.model.weights_float64", "non-finite weight");
  for (let i = 1; i < columns.length; i++)
    if ((columns[i] ?? 0) <= (columns[i - 1] ?? 0)) d.fail("$.model.columns_int32", "not strictly increasing");

  return { labels, columns, weights, bias };
}
