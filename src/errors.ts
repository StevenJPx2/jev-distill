export type ErrorCode =
  | "invalid_contract"
  | "invalid_dataset"
  | "invalid_state"
  | "invalid_ledger"
  | "invalid_artifact"
  | "approval_required"
  | "budget_exhausted"
  | "teacher_unavailable"
  | "training_failed"
  | "usage"
  | "io";

/** Expected, user-facing failure. Anything else escaping the CLI is a defect. */
export class DistillError extends Error {
  readonly code: ErrorCode;

  constructor(code: ErrorCode, message: string) {
    super(message);
    this.name = "DistillError";
    this.code = code;
  }
}
