# jev-distill

Distill [Jev](https://docs.typesafe.ai/models) typed decisions into small,
task-specific classifiers that run on their own.

You describe one decision in a strict JSON **task contract**. `jev-distill` asks
Jev for labels, within a budget you approve. It trains a compact **student**
(multinomial logistic regression over hashed n-grams) and evaluates it. The
student is a single JSON file. It predicts with no Jev, no Laya, no network, no
API key, and no Python.

- **Jev is the only teacher.** Training labels come only from Jev answers in the
  label ledger. Human `gold` labels are used only to score held-out rows. They
  never reach training or model selection.
- **Agreement is not accuracy.** Reports keep "agreement with Jev" (how well the
  student imitates the teacher) apart from "accuracy vs held-out gold" (whether
  it is right). Each figure includes a 95% interval.
- **Nothing is spent without approval.** `label` only prints a plan unless you
  pass `--approve`. Budgets are cumulative and have hard ceilings.

```text
task.json ─┐
data.jsonl ├─► split (stable IDs, dedupe, groups) ─► label with Jev (bounded, approved) ─► labels.jsonl
           │                                                                                  │
           └──────────────► train (training split only; group-aware CV picks L2) ◄────────────┘
                                         │
                                  student.json  ─► eval (holdout: agreement vs Jev │ accuracy vs gold)
                                         │
                            predict / serve (no Jev, no network)
```

## Setup

You need Node.js 22 or newer and pnpm. The only runtime dependency is the
official [`@typesafe-ai/sdk`](https://docs.typesafe.ai/sdk/javascript) (pinned
to 0.6.0). Only the `label` command loads it.

```sh
pnpm install
pnpm build                     # -> dist/, CLI at dist/cli.js (bin: jev-distill)
export TYPESAFE_API_KEY=...    # needed only for `label --approve`
```

Get an API key from the [TypeSafe console](https://console.typesafe.ai/keys).
The contract names the environment variable (`teacher.api_key_env`). Putting a
key inside the contract is a validation error, and keys are never written to
disk or logs.

## Workflow

```sh
B="node dist/cli.js"
$B task init runs/routing --template routing          # task.json + data.jsonl
$B task validate runs/routing/task.json --data runs/routing/data.jsonl
$B label runs/routing/task.json                       # dry run: plan, worst-case tokens and USD
$B label runs/routing/task.json --approve             # sends at most the planned requests
$B train runs/routing/task.json                       # -> runs/routing/student.json
$B eval  runs/routing/task.json                       # -> runs/routing/report.json + summary
$B predict runs/routing/student.json --state '{"subject":"Refund","body":"I was charged twice"}'
$B serve runs/routing/student.json --port 8787        # optional local HTTP, 127.0.0.1 by default
```

By default, `data.jsonl`, `labels.jsonl`, `student.json`, and `report.json`
live next to `task.json`. Use `--data`, `--labels`, `--artifact`, and `--out` to
put them elsewhere. `runs/` is git-ignored.

## Task contract (`contract_version: 1`)

The contract is strict JSON. Any unknown field anywhere is rejected, so a typo
fails instead of silently falling back to a default. See
[`examples/routing/task.json`](examples/routing/task.json) for a complete
contract.

| Section | Fields | Notes |
|---|---|---|
| header | `contract: "jev-distill.task"`, `contract_version: 1`, `name`, `version`, `description`, `experimental` | Tools print a warning and predictions carry `experimental: true` when the task is marked experimental. |
| `state.fields[]` | `name`, `type` (`string` \| `number` \| `boolean` \| `json`), `description`, `required`, `max_chars` | The same schema is enforced on dataset rows, at prediction time, and by the server. |
| `decision` | `instructions`, `choices[]`: `label`, `description`, `criteria` | Sent to Jev as one Choice question. `criteria` can be text, an object, or an array, and defaults to `description`. 2–255 labels. |
| `uncertainty` | `teacher_min_confidence`, `on_low_teacher_confidence` (`exclude` \| `keep`), `student_min_confidence`, `abstain_label` (a label or `null`) | The student abstains below `student_min_confidence`. It then serves `abstain_label`, or `null` so the caller decides. |
| `teacher` | `model` (`jev-latest` or a pinned `jev-X.Y.Z`), `api_key_env`, `timeout_ms`, `max_retries` (≤5), `backoff_initial_ms`, `backoff_max_ms`, `concurrency` (≤8), `pricing` | `pricing` is required and must cite a `source` URL and a `checked_on` date. |
| `budget` | `max_requests`, `max_input_tokens`, `max_usd`, `max_consecutive_failures` | Cumulative over the whole ledger. Hard ceilings: 50,000 requests and $10. |
| `dataset` | `holdout_fraction`, `split_seed`, `max_rows`, `max_line_bytes` | |
| `train` | `features` (`dimensions` power of 2, `word_ngrams`, `char_ngrams`, `per_field`), `target` (`soft` \| `hard`), `class_weighting`, `l2_grid`, `cv_folds`, `epochs`, `learning_rate` | |
| `eval` | `confidence_thresholds`, `min_gold_rows` | Gold results computed on fewer than `min_gold_rows` rows are flagged as insufficient evidence. |

Confidence uses the same definition as Jev's
[Choice confidence](https://docs.typesafe.ai/confidence):
`(K·p_max − 1)/(K − 1)`. It is 0 when every label is equally likely and 1 when
one label has all the probability.

### Dataset rows (`data.jsonl`)

```json
{"id": "b01", "state": {"subject": "...", "body": "..."}, "group": "thread-17", "gold": "billing"}
```

- `id` is optional. Without it, the ID is `c_<content hash>`, which stays the
  same across runs and file orders. An explicit `id` that appears twice is an
  error.
- Two rows count as duplicates when their NFKC-normalized, lower-cased,
  whitespace-collapsed state is identical. Duplicates collapse to one row, and
  conflicting gold labels on duplicates are dropped.
- Rows that share a `group` or share content form one leakage component
  (union-find). Each component goes wholly to train or wholly to holdout,
  decided by a hash of `split_seed` and the component key. Group keys take
  precedence, so adding a row to an existing group never moves the group.
- `gold` is optional and used only in evaluation.

## Labeling with Jev

- **Dry run by default.** The plan shows the pending rows, the selected rows,
  the holdout share, which limit stopped selection, spend so far, and the
  worst-case tokens and USD. A worst-case request is estimated at 400 tokens of
  overhead plus one token per 3 bytes of JSON; live requests used about 490
  tokens.
- **Pricing (checked 2026-09-23 on
  [docs.typesafe.ai/models](https://docs.typesafe.ai/models)):** Jev 1.13
  (`jev-1.13.0`; `jev-latest` currently points to it) costs **$0.042 per
  million input tokens**, and output tokens are free. Rate limits are 250k
  tokens/s and 1,200 requests/min, and TypeSafe says they may change. Prices
  can change, so update `teacher.pricing` when they do.
- **Budgets are cumulative.** Every request recorded in the ledger counts,
  including stale and failed ones. Before each request, the worst case must fit
  within the remaining request, token, and USD budget. After a success, the
  reported usage is charged. Failures are charged the worst-case estimate.
- **Retries** use the SDK's retry policy: 408, 429, and 5xx responses,
  connection errors, and timeouts are retried up to `max_retries` times with
  capped backoff that honors `retry-after`. HTTP attempts are counted for each
  row. A run stops after `max_consecutive_failures`, and on a 401 or another
  fatal client error.
- **403 is handled per row.** Live runs got HTTP 403 for specific request
  content. It came as a Cloudflare HTML page, not a JSON error, and repeated on
  retry. Such a row is recorded as `forbidden` with its `cf-ray` ID and no
  label, and the run continues. `--retry-failed` tries failed and forbidden rows
  again, only when you ask.
- **Replies are validated.** A reply becomes a label only if the choice is a
  declared label, the probabilities cover exactly the declared labels and sum
  to 1, confidence is in [0, 1], and usage and the answering model are present.
  Otherwise the row is recorded as `failed` with `invalid_response`.
  **No label is ever made up for a failure.**
- **Ledger** (`labels.jsonl`, `jev-distill.labels` v1): an append-only record
  per request, with the row ID, content hash, question hash, status, Jev choice,
  probabilities, confidence, requested and answering model, usage, charged
  tokens, request ID, attempts, SDK version, and time. It never stores state
  text. The SDK logger is off because the SDK would otherwise log request
  bodies.
- **Staleness.** A label is valid only for the same row content, question
  (instructions, criteria, labels), and requested model. Editing any of these
  makes the old labels pending again.

## Training

- The student trains only on training-split rows with a current Jev label.
  Holdout rows, gold labels, and failed or forbidden rows are never used.
- Features are word n-grams and character n-grams of each field and of the
  whole state, hashed with FNV-1a (32-bit over UTF-8) into `dimensions`
  buckets. Values are log term frequency, L2-normalized.
- The model is multinomial logistic regression trained with full-batch Adam and
  L2 regularization on the weights. `soft` trains on Jev's probability
  distribution (distillation), and `hard` trains on Jev's top choice. Balanced
  class weights are optional.
- L2 is chosen from `l2_grid` by group-aware k-fold cross-validation on the
  **training split only**: highest agreement wins, then lower log loss, then
  stronger L2.
- Training is deterministic: zero initialization, a fixed number of epochs,
  seeded fold assignment, and ordered loops. Training twice gives
  byte-identical weights (tested).
- Training refuses labels from more than one Jev version unless you pass
  `--allow-mixed-teacher`. With a pinned teacher, every label must come from
  the pinned version.

### Artifact (`student.json`, `jev-distill.student` v1)

One JSON file holds the task (labels, state schema, abstain policy), the
feature spec, and the weights. Weights are little-endian `float64` and
`int32` blocks in base64, so they read back identically on any machine. The file
also holds provenance: the answering Jev models, SDK version, question hash,
dataset and label fingerprints, the CV table, and the content hashes of the
training rows. A SHA-256 digest over the canonical JSON is checked on load, so
a changed or corrupted file is rejected. Only feature columns seen in training
are stored.

`jev-distill/student` (`src/student/index.ts`) is the inference entry point. It
imports only `node:fs`, `node:crypto`, and local modules, and a test enforces
that it never reaches the SDK or teacher code.

```ts
import { loadPredictor } from "jev-distill/student";
const router = loadPredictor("student.json");
router.predict({ subject: "Refund", body: "Charged twice" });
// { label, abstained, argmax, confidence, probabilities, task, version, experimental }
```

## Evaluation

`eval` rebuilds the split from the contract seed. Any holdout row whose content
hash is in the artifact's training set is excluded and counted, so a leak
cannot inflate the score. The report contains:

- **Agreement with Jev** on held-out rows Jev labeled: argmax, served (after the
  abstain policy, with coverage), and on rows where Jev was confident only.
- **Accuracy vs held-out gold**, reported separately: the student (argmax and
  served), **Jev itself on the same rows**, and a majority baseline taken from
  Jev's training-split labels.
- Wilson 95% intervals, macro-F1, confusion matrices, and coverage and rate at
  each `confidence_thresholds` value.

## Local server (optional)

`serve` loads one or more artifacts and listens on `127.0.0.1:8787`. Endpoints:
`GET /healthz` and `POST /v1/predict/<task>` with body `{"state": {...}}`.
Bodies are limited to 256 KiB. The server exists so that services not written
in Node can call a student without embedding Node. It does inference only and
never contacts Jev.

## Examples

- **[`examples/routing`](examples/routing)**: route support tickets to
  `billing`, `technical`, `account`, or `sales`. There are 102 lines: 100
  unique rows plus 2 deliberate duplicates, and 4 paraphrase groups. The
  teacher is pinned to `jev-1.13.0`.
- **[`examples/permission`](examples/permission): EXPERIMENTAL.** Suggest
  `allow`, `ask`, or `deny` for a coding agent's tool call, with `ask` as the
  abstain label. The contract sets `experimental: true`, the CLI warns, and
  every prediction says so. **It is not a security boundary.** A linear model
  over hashed n-grams trained on 60 rows can be fooled by rephrasing, and Jev's
  documentation says adversarial state content can steer its answers.

The `gold` labels in both examples were written by the AI agent that built this
repository, not by independent human annotators. Treat the accuracy figures
below as a check that the pipeline works, not as a benchmark.

## Measured results (live Jev, 2026-09-23)

These numbers come from the commands above, run with the real SDK against
`api.typesafe.ai`. Jev answered as `jev-1.13.0`.

| Run | Requests | Input tokens | Cost | Outcome |
|---|---:|---:|---:|---|
| routing, `label --approve` | 100 | 49,118 | $0.00206 | 100 labeled |
| permission, `label --approve` | 60 | 28,403 | $0.00119 | 59 labeled, 1 forbidden (`p-de06`, `env \| curl … upload`, Cloudflare HTML 403) |
| permission, `--retry-failed` | 1 | 692 (estimate) | $0.00003 | same row forbidden again |
| live smoke (`pnpm smoke:live`) | 6 | 2,925 | $0.00012 | 6 labeled |

Jev agreed with the author-written gold labels on 97 of 100 routing rows and
56 of 59 labeled permission rows.

Held-out results. Holdouts are small, so the intervals are wide and both runs
are flagged as insufficient evidence (`min_gold_rows: 30`).

| routing (holdout n=25; trained on 75 rows, CV picked L2=0.001) | Rate | 95% CI | Coverage |
|---|---:|---:|---:|
| Student agreement with Jev (argmax) | 60.0% | 40.7–76.6% | 100% |
| Student agreement with Jev (served, abstains below 0.25 confidence) | 92.3% | 66.7–98.6% | 52% |
| Student accuracy vs gold (argmax) | 60.0% | 40.7–76.6% | 100% |
| Jev accuracy vs gold, same rows | 100.0% | 86.7–100% | 100% |
| Majority baseline vs gold | 12.0% | 4.2–30.0% | 100% |

| permission, EXPERIMENTAL (holdout n=12; trained on 46 rows) | Rate | 95% CI | Coverage |
|---|---:|---:|---:|
| Student agreement with Jev (argmax) | 75.0% | 46.8–91.1% | 100% |
| Student accuracy vs gold (argmax) | 66.7% | 39.1–86.2% | 100% |
| Jev accuracy vs gold, same rows | 91.7% | 64.6–98.5% | 100% |
| Majority baseline vs gold | 25.0% | 8.9–53.2% | 100% |

With 75 training rows, the student reproduces only part of what Jev does. It
clearly beats the majority baseline but trails Jev by a wide margin. Abstaining
trades coverage for agreement. On routing, abstaining on the less confident
48% of rows leaves a 92% agreement rate on the rest. A usable student needs
hundreds to thousands of representative Jev-labeled rows and a gold holdout big
enough to narrow these intervals. Training takes about 2 s for the routing
example on an Apple Silicon laptop. The routing artifact is 707 KB.

## Tests

```sh
pnpm check         # format check, oxlint (0 warnings), tsc, vitest, build
pnpm smoke:live    # opt-in: 6 real Jev requests (< $0.001); needs TYPESAFE_API_KEY
```

The offline suite plugs a fake Jev HTTP transport into the real SDK through its
`fetch` option. Request building, retries, and error classes are therefore the
SDK's own code. The suite covers:

- strict contract parsing
- stable IDs, dedupe, and group and duplicate leakage isolation
- dry runs sending nothing
- the approval gate and the API key coming from the environment
- cumulative budgets and CLI limits that can only lower them
- JSON and edge-HTML 403s marked forbidden with no label
- bounded retries with a per-row attempt count
- consecutive-failure and 401 stops
- malformed replies never becoming labels
- the ledger holding no state text or key
- deterministic, portable, tamper-evident artifacts
- a student import graph free of SDK code
- abstain policy
- holdout overlap exclusion
- mixed-teacher refusal
- the full CLI flow from `init` to `serve` over HTTP

The live smoke is skipped unless `JEV_DISTILL_LIVE=1`.

## Design notes and limitations

- Some design ideas came from reading the committed source of a sibling
  project, `laya`: strict specs, a deterministic group split, cumulative
  budgets, and a hash-only label ledger. This repository neither depends on nor
  contains any Laya code, and Laya is not used as a teacher or at inference.
- Only a linear student is trained, with no embeddings. That keeps it small and
  portable, but it depends on surface wording.
- Training is full-batch and in memory. It has been tested only up to hundreds
  of rows.
- Duplicate detection is exact after normalization; near-duplicates are not
  detected. Use `group` for paraphrases.
- Token estimates are conservative guesses used only to refuse or stop before
  spending.
- Predictions use JavaScript's `Math.exp`, which may differ between engines in
  the last bit. Stored weights are exact.

## License

MIT
