import type { SparseVector } from "../student/features.js";
import { type LinearModel, softmaxInPlace } from "../student/model.js";

export interface TrainingSet {
  readonly rows: readonly SparseVector[];
  /** Target distribution per row over `labels` (one-hot for hard targets). */
  readonly targets: readonly Float64Array[];
  readonly weights: Float64Array;
  readonly labels: readonly string[];
}

export interface FitOptions {
  readonly l2: number;
  readonly epochs: number;
  readonly learningRate: number;
}

export interface Fit {
  readonly model: LinearModel;
  readonly finalLoss: number;
}

/** Compact local columns: only buckets present in training rows can receive weight. */
interface Local {
  readonly columns: Int32Array;
  readonly rows: readonly { readonly cols: Int32Array; readonly values: Float64Array }[];
}

function localize(rows: readonly SparseVector[]): Local {
  const buckets = [...new Set(rows.flatMap((row) => [...row.indices]))].sort((a, b) => a - b);
  const position = new Map(buckets.map((bucket, i) => [bucket, i]));

  return {
    columns: Int32Array.from(buckets),
    rows: rows.map((row) => ({ cols: Int32Array.from(row.indices, (bucket) => position.get(bucket) ?? 0), values: row.values })),
  };
}

/**
 * Full-batch Adam on weighted soft-target cross-entropy with L2 on weights
 * (not biases). Zero init, fixed epoch count, and ordered loops make the
 * result deterministic for a given input.
 */
export function fitLogistic(set: TrainingSet, options: FitOptions): Fit {
  const k = set.labels.length;
  const local = localize(set.rows);
  const biasOffset = local.columns.length * k;
  const params = new Float64Array(biasOffset + k);
  const adam = { m: new Float64Array(params.length), v: new Float64Array(params.length), t: 0 };
  const totalWeight = set.weights.reduce((sum, w) => sum + w, 0);
  let loss = Number.NaN;

  for (let epoch = 0; epoch < options.epochs; epoch++) {
    const grad = new Float64Array(params.length);
    loss = accumulate(set, local, params, grad, totalWeight);

    for (let i = 0; i < biasOffset; i++) {
      const w = params[i] ?? 0;
      loss += 0.5 * options.l2 * w * w;
      grad[i] = (grad[i] ?? 0) + options.l2 * w;
    }

    adamStep(params, grad, adam, options.learningRate);
  }

  return {
    model: { labels: [...set.labels], columns: local.columns, weights: params.slice(0, biasOffset), bias: params.slice(biasOffset) },
    finalLoss: loss,
  };
}

function accumulate(set: TrainingSet, local: Local, params: Float64Array, grad: Float64Array, totalWeight: number): number {
  const k = set.labels.length;
  const biasOffset = local.columns.length * k;
  let loss = 0;

  local.rows.forEach((row, r) => {
    const logits = params.slice(biasOffset, biasOffset + k);
    for (let p = 0; p < row.cols.length; p++) {
      const base = (row.cols[p] ?? 0) * k;
      const value = row.values[p] ?? 0;
      for (let c = 0; c < k; c++) logits[c] = (logits[c] ?? 0) + (params[base + c] ?? 0) * value;
    }

    const probs = softmaxInPlace(logits);
    const target = set.targets[r] ?? new Float64Array(k);
    const scale = (set.weights[r] ?? 0) / totalWeight;

    for (let c = 0; c < k; c++) {
      const q = target[c] ?? 0;
      if (q > 0) loss -= scale * q * Math.log(Math.max(probs[c] ?? 0, 1e-12));

      const g = scale * ((probs[c] ?? 0) - q);
      grad[biasOffset + c] = (grad[biasOffset + c] ?? 0) + g;
      for (let p = 0; p < row.cols.length; p++) {
        const index = (row.cols[p] ?? 0) * k + c;
        grad[index] = (grad[index] ?? 0) + g * (row.values[p] ?? 0);
      }
    }
  });

  return loss;
}

function adamStep(params: Float64Array, grad: Float64Array, state: { m: Float64Array; v: Float64Array; t: number }, rate: number): void {
  const [beta1, beta2, epsilon] = [0.9, 0.999, 1e-8];
  state.t += 1;
  const c1 = 1 - beta1 ** state.t;
  const c2 = 1 - beta2 ** state.t;

  for (let i = 0; i < params.length; i++) {
    const g = grad[i] ?? 0;
    const m = beta1 * (state.m[i] ?? 0) + (1 - beta1) * g;
    const v = beta2 * (state.v[i] ?? 0) + (1 - beta2) * g * g;
    state.m[i] = m;
    state.v[i] = v;
    params[i] = (params[i] ?? 0) - (rate * (m / c1)) / (Math.sqrt(v / c2) + epsilon);
  }
}
