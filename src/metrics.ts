export interface Interval {
  readonly low: number;
  readonly high: number;
}

export interface LabelMetrics {
  readonly n: number;
  readonly correct: number;
  readonly rate: number | null;
  /** Wilson 95% interval; wide intervals mean the rate is weak evidence. */
  readonly interval95: Interval | null;
  readonly macro_f1: number | null;
  /** confusion[reference][predicted] */
  readonly confusion: Readonly<Record<string, Readonly<Record<string, number>>>>;
}

export function wilson(correct: number, n: number): Interval | null {
  if (n === 0) return null;

  const z = 1.959964;
  const p = correct / n;
  const denom = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;

  return { low: Math.max(0, center - half), high: Math.min(1, center + half) };
}

/** Compare predictions with reference labels; `null` predictions (abstentions) must be filtered by the caller. */
export function compare(
  labels: readonly string[],
  pairs: readonly { readonly reference: string; readonly predicted: string }[],
): LabelMetrics {
  const confusion: Record<string, Record<string, number>> = Object.fromEntries(
    labels.map((reference) => [reference, Object.fromEntries(labels.map((predicted) => [predicted, 0]))]),
  );

  let correct = 0;
  for (const { reference, predicted } of pairs) {
    const row = confusion[reference];
    if (row) row[predicted] = (row[predicted] ?? 0) + 1;
    if (reference === predicted) correct += 1;
  }

  return {
    n: pairs.length,
    correct,
    rate: pairs.length === 0 ? null : correct / pairs.length,
    interval95: wilson(correct, pairs.length),
    macro_f1: pairs.length === 0 ? null : macroF1(labels, confusion),
    confusion,
  };
}

/** Macro-F1 over labels that occur in the reference or the predictions. */
function macroF1(labels: readonly string[], confusion: Record<string, Record<string, number>>): number {
  const scores: number[] = [];

  for (const label of labels) {
    const tp = confusion[label]?.[label] ?? 0;
    const actual = Object.values(confusion[label] ?? {}).reduce((sum, v) => sum + v, 0);
    const predicted = labels.reduce((sum, ref) => sum + (confusion[ref]?.[label] ?? 0), 0);
    if (actual === 0 && predicted === 0) continue;

    scores.push(actual + predicted === 0 ? 0 : (2 * tp) / (actual + predicted));
  }

  return scores.length === 0 ? 0 : scores.reduce((sum, v) => sum + v, 0) / scores.length;
}
