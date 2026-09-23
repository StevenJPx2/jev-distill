import { labelsOf, type TaskContract } from "../contract.js";
import { canonicalJson, type Entry, type JsonObject, type JsonValue, sha256Hex } from "../json.js";

export const QUESTION_KEY = "decision";

export interface JevChoiceQuestion {
  readonly type: "choice";
  readonly instructions: Entry;
  readonly criteria: Record<string, Entry>;
}

export function jevQuestion(contract: TaskContract): JevChoiceQuestion {
  return {
    type: "choice",
    instructions: contract.decision.instructions,
    criteria: Object.fromEntries(contract.decision.choices.map((choice) => [choice.label, choice.criteria])),
  };
}

/**
 * Labels are only valid for the exact question and requested model that
 * produced them; editing either makes earlier labels stale.
 */
export function questionHash(contract: TaskContract): string {
  const question = jevQuestion(contract);

  return sha256Hex(
    canonicalJson({ model: contract.teacher.model, labels: labelsOf(contract), question: question as unknown as JsonObject }),
  );
}

/** Conservative worst-case input tokens for one request: fixed overhead plus ~1 token per 3 UTF-8 bytes. */
export function estimateInputTokens(state: JsonValue, question: JevChoiceQuestion): number {
  const bytes = Buffer.byteLength(JSON.stringify({ state, questions: { [QUESTION_KEY]: question } }));

  return 400 + Math.ceil(bytes / 3);
}
