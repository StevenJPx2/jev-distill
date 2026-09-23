import type { FeatureOptions } from "../contract.js";
import type { JsonValue } from "../json.js";
import { fieldText, normalizeText } from "../state.js";

/** Everything needed to reproduce featurization on another machine. */
export interface FeatureSpec extends FeatureOptions {
  readonly kind: "hashed-ngrams";
  readonly version: 1;
  readonly hash: "fnv1a32-utf8";
  readonly normalization: "nfkc-lowercase-collapse-whitespace";
  readonly weighting: "log1p-tf-l2";
}

export interface SparseVector {
  /** Strictly increasing bucket indices. */
  readonly indices: Int32Array;
  readonly values: Float64Array;
}

const encoder = new TextEncoder();
const TOKEN = /[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu;

export function featureSpec(options: FeatureOptions): FeatureSpec {
  return {
    kind: "hashed-ngrams",
    version: 1,
    hash: "fnv1a32-utf8",
    normalization: "nfkc-lowercase-collapse-whitespace",
    weighting: "log1p-tf-l2",
    dimensions: options.dimensions,
    word_ngrams: options.word_ngrams,
    char_ngrams: options.char_ngrams,
    per_field: options.per_field,
  };
}

export function fnv1a32(text: string): number {
  let hash = 0x811c9dc5;

  for (const byte of encoder.encode(text)) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193);
  }

  return hash >>> 0;
}

/** Hashed word and character n-grams per field (and globally), log-TF weighted, L2-normalized. */
export function featurize(spec: FeatureSpec, fieldNames: readonly string[], state: Readonly<Record<string, JsonValue>>): SparseVector {
  const counts = new Map<number, number>();
  const add = (feature: string): void => {
    const bucket = fnv1a32(feature) & (spec.dimensions - 1);
    counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
  };

  for (const name of fieldNames) {
    const value = state[name];
    if (value === undefined) {
      add(`missing|${name}`);
      continue;
    }

    const text = normalizeText(fieldText(value));
    for (const feature of textFeatures(spec, text)) {
      add(feature);
      if (spec.per_field) add(`${name}|${feature}`);
    }
  }

  const indices = Int32Array.from([...counts.keys()].sort((a, b) => a - b));
  const values = Float64Array.from(indices, (index) => Math.log1p(counts.get(index) ?? 0));
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));

  if (norm > 0) for (let i = 0; i < values.length; i++) values[i] = (values[i] ?? 0) / norm;

  return { indices, values };
}

function textFeatures(spec: FeatureSpec, text: string): string[] {
  const features: string[] = [];
  const tokens = text.match(TOKEN) ?? [];

  for (let n = 1; n <= spec.word_ngrams; n++) {
    for (let i = 0; i + n <= tokens.length; i++) features.push(`w${n}:${tokens.slice(i, i + n).join(" ")}`);
  }

  if (spec.char_ngrams !== null) {
    const [lo, hi] = spec.char_ngrams;
    const chars = [...` ${text} `];

    for (let n = lo; n <= hi; n++) {
      for (let i = 0; i + n <= chars.length; i++) features.push(`c${n}:${chars.slice(i, i + n).join("")}`);
    }
  }

  return features;
}
