import type { TrainOptions } from "../contract.js";
import { unitInterval } from "../json.js";
import { argmax, probabilities } from "../student/model.js";
import { fitLogistic, type TrainingSet } from "./optimizer.js";

export interface CvScore {
  readonly l2: number;
  /** Mean held-fold agreement with Jev's top label (not accuracy). */
  readonly agreement: number;
  /** Mean held-fold cross-entropy against the training target. */
  readonly log_loss: number;
}

export interface Selection {
  readonly l2: number;
  readonly folds: number;
  readonly scores: readonly CvScore[];
  readonly note: string | null;
}

/**
 * Group-aware k-fold CV over the training split only. Folds are assigned per
 * leakage component, so related rows never sit on both sides of a fold.
 * Highest agreement wins; ties prefer lower log loss, then stronger L2.
 */
export function selectL2(set: TrainingSet, components: readonly string[], options: TrainOptions, seed: string): Selection {
  const unique = [...new Set(components)].sort(
    (a, b) => unitInterval(`${seed}\u0000cv\u0000${a}`) - unitInterval(`${seed}\u0000cv\u0000${b}`),
  );
  const folds = Math.min(options.cv_folds, unique.length);
  const strongest = options.l2_grid[options.l2_grid.length - 1] ?? 1;

  if (options.l2_grid.length === 1)
    return { l2: options.l2_grid[0] ?? strongest, folds: 0, scores: [], note: "single l2 value; CV skipped" };
  if (folds < 2) return { l2: strongest, folds: 0, scores: [], note: "fewer than 2 components; CV skipped, strongest l2 used" };

  const foldOf = new Map(unique.map((component, i) => [component, i % folds]));
  const scores = options.l2_grid.map((l2) =>
    scoreL2(
      set,
      components.map((c) => foldOf.get(c) ?? 0),
      folds,
      l2,
      options,
    ),
  );
  const best = [...scores].sort((a, b) => b.agreement - a.agreement || a.log_loss - b.log_loss || b.l2 - a.l2)[0];

  return { l2: best?.l2 ?? strongest, folds, scores, note: null };
}

function scoreL2(set: TrainingSet, foldOfRow: readonly number[], folds: number, l2: number, options: TrainOptions): CvScore {
  let agree = 0;
  let loss = 0;
  let count = 0;

  for (let fold = 0; fold < folds; fold++) {
    const trainIdx = foldOfRow.flatMap((f, i) => (f === fold ? [] : [i]));
    const testIdx = foldOfRow.flatMap((f, i) => (f === fold ? [i] : []));
    if (trainIdx.length === 0 || testIdx.length === 0) continue;

    const { model } = fitLogistic(subset(set, trainIdx), { l2, epochs: options.epochs, learningRate: options.learning_rate });

    for (const i of testIdx) {
      const probs = probabilities(model, set.rows[i] ?? { indices: new Int32Array(), values: new Float64Array() });
      const target = set.targets[i] ?? new Float64Array();

      if (argmax(probs) === argmax(target)) agree += 1;
      target.forEach((q, c) => {
        if (q > 0) loss -= q * Math.log(Math.max(probs[c] ?? 0, 1e-12));
      });
      count += 1;
    }
  }

  return { l2, agreement: count === 0 ? 0 : agree / count, log_loss: count === 0 ? Infinity : loss / count };
}

function subset(set: TrainingSet, indices: readonly number[]): TrainingSet {
  return {
    labels: set.labels,
    rows: indices.map((i) => set.rows[i]).filter((row) => row !== undefined),
    targets: indices.map((i) => set.targets[i]).filter((target) => target !== undefined),
    weights: Float64Array.from(indices, (i) => set.weights[i] ?? 0),
  };
}
