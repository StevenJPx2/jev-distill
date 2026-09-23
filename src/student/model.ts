import type { SparseVector } from "./features.js";

/**
 * Multinomial logistic regression over hashed features. Only feature buckets
 * seen in training can have non-zero weights, so just those columns are kept.
 */
export interface LinearModel {
  readonly labels: readonly string[];
  /** Strictly increasing feature buckets with stored weights. */
  readonly columns: Int32Array;
  /** Row-major `[columns.length x labels.length]`. */
  readonly weights: Float64Array;
  readonly bias: Float64Array;
}

export function softmaxInPlace(logits: Float64Array): Float64Array {
  let top = -Infinity;
  for (const value of logits) top = Math.max(top, value);

  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const value = Math.exp((logits[i] ?? 0) - top);
    logits[i] = value;
    sum += value;
  }

  for (let i = 0; i < logits.length; i++) logits[i] = (logits[i] ?? 0) / sum;

  return logits;
}

function findColumn(columns: Int32Array, bucket: number): number {
  let lo = 0;
  let hi = columns.length - 1;

  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const value = columns[mid] ?? 0;
    if (value === bucket) return mid;
    if (value < bucket) lo = mid + 1;
    else hi = mid - 1;
  }

  return -1;
}

export function probabilities(model: LinearModel, x: SparseVector): Float64Array {
  const k = model.labels.length;
  const logits = Float64Array.from(model.bias);

  for (let p = 0; p < x.indices.length; p++) {
    const column = findColumn(model.columns, x.indices[p] ?? -1);
    if (column < 0) continue;

    const value = x.values[p] ?? 0;
    for (let c = 0; c < k; c++) logits[c] = (logits[c] ?? 0) + (model.weights[column * k + c] ?? 0) * value;
  }

  return softmaxInPlace(logits);
}

export function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if ((values[i] ?? 0) > (values[best] ?? 0)) best = i;

  return best;
}

/**
 * Same definition Jev documents for Choice confidence: 0 for a uniform
 * distribution, 1 when all mass is on one option.
 */
export function choiceConfidence(probs: ArrayLike<number>): number {
  const k = probs.length;
  const peak = probs[argmax(probs)] ?? 0;

  return Math.max(0, Math.min(1, (k * peak - 1) / (k - 1)));
}
