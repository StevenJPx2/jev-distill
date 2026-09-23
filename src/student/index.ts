/**
 * Inference entry point. Imports nothing from the Jev SDK or the network:
 * a student artifact plus this module is all a consumer needs.
 */
import { validateState } from "../state.js";
import { loadArtifact, type StudentArtifact } from "./artifact.js";
import { featurize } from "./features.js";
import { argmax, choiceConfidence, probabilities } from "./model.js";

export interface Prediction {
  readonly task: string;
  readonly version: string;
  readonly experimental: boolean;
  /** Served label after the abstain policy; `null` when abstaining without an abstain label. */
  readonly label: string | null;
  readonly abstained: boolean;
  /** Highest-probability label before the abstain policy. */
  readonly argmax: string;
  readonly confidence: number;
  readonly probabilities: Readonly<Record<string, number>>;
}

export interface Predictor {
  readonly artifact: StudentArtifact;
  predict(rawState: unknown): Prediction;
}

export function createPredictor(artifact: StudentArtifact): Predictor {
  const { task, features, model } = artifact;
  const fieldNames = task.state_fields.map((field) => field.name);

  return {
    artifact,
    predict(rawState: unknown): Prediction {
      const state = validateState(task.state_fields, rawState, "input");
      const probs = probabilities(model, featurize(features, fieldNames, state));
      const top = model.labels[argmax(probs)] ?? "";
      const confidence = choiceConfidence(probs);
      const abstained = confidence < task.uncertainty.student_min_confidence;

      return {
        task: task.name,
        version: task.version,
        experimental: task.experimental,
        label: abstained ? task.uncertainty.abstain_label : top,
        abstained,
        argmax: top,
        confidence,
        probabilities: Object.fromEntries(model.labels.map((label, i) => [label, probs[i] ?? 0])),
      };
    },
  };
}

export function loadPredictor(path: string): Predictor {
  return createPredictor(loadArtifact(path));
}

export { loadArtifact, parseArtifact, type StudentArtifact } from "./artifact.js";
