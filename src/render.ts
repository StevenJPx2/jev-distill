import type { EvalReport, Served } from "./eval.js";
import type { LabelMetrics } from "./metrics.js";

const pct = (value: number | null): string => (value === null ? "n/a" : `${(value * 100).toFixed(1)}%`);

function row(name: string, m: LabelMetrics, coverage?: number | null): string {
  const interval = m.interval95 ? `${pct(m.interval95.low)}–${pct(m.interval95.high)}` : "n/a";
  const f1 = m.macro_f1 === null ? "n/a" : m.macro_f1.toFixed(3);

  return `| ${name} | ${m.n} | ${pct(m.rate)} | ${interval} | ${f1} | ${coverage === undefined ? "100%" : pct(coverage)} |`;
}

function servedRow(name: string, served: Served): string {
  return row(name, served.metrics, served.coverage);
}

export function renderEval(report: EvalReport): string {
  const h = report.holdout;
  const t = report.teacher_agreement;
  const lines = [
    `# ${report.task} ${report.task_version}${report.experimental ? " (EXPERIMENTAL)" : ""}`,
    "",
    `Holdout rows: ${h.rows} (Jev-labeled ${h.teacher_labeled}, forbidden ${h.teacher_forbidden}, failed ${h.teacher_failed}, unlabeled ${h.teacher_unlabeled}, gold ${h.with_gold}; excluded as trained-on ${h.excluded_trained_overlap})`,
    "",
    "## Agreement with Jev (teacher imitation, not accuracy)",
    "",
    "| Measure | n | Agreement | 95% CI | Macro-F1 | Coverage |",
    "|---|---:|---:|---:|---:|---:|",
    row("Student argmax vs Jev", t.argmax),
    servedRow("Student served vs Jev", t.served),
    row("Student argmax vs confident Jev", t.teacher_confident_only),
    "",
  ];

  const g = report.gold;
  if (g === null) {
    lines.push("## Accuracy vs gold", "", "No held-out gold labels; accuracy is unmeasured.", "");
  } else {
    lines.push(
      `## Accuracy vs held-out gold${g.sufficient_evidence ? "" : " (INSUFFICIENT EVIDENCE: too few gold rows)"}`,
      "",
      "| Measure | n | Accuracy | 95% CI | Macro-F1 | Coverage |",
      "|---|---:|---:|---:|---:|---:|",
      row("Student argmax", g.student_argmax),
      servedRow("Student served", g.student_served),
      row("Jev (teacher) on same rows", g.jev_on_same_rows),
      row(`Majority baseline (${g.majority_baseline.label ?? "none"})`, g.majority_baseline),
      "",
    );
  }

  return `${lines.join("\n")}\n`;
}
