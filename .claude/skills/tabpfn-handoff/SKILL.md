---
name: tabpfn-handoff
description: Delegate any tabular prediction task (classification, regression) to TabPFN via the Prior Labs MCP server, after doing the data work yourself. Use whenever the user asks to predict, classify, or score anything from tabular data — churn, risk, prices, outcomes — including scoring a pasted row or CSV snippet of new data. Runs an XGBoost baseline only when the user asks for a comparison; writes structured markdown reports only when the user asks for one.
---

# TabPFN handoff

You are the data scientist; TabPFN is the model. You do everything a good analyst does
*around* the model — and refuse to do the model's job.
TabPFN is designed to make predictions (classification and regression)
over structured data (like CSVs and tables) and fits in one forward pass with no
hyperparameter search. Any task that is inferred to require a calibrated prediction
should be handed off to TabPFN — never train a local model as the answer.

**Working files:** stage everything in `reports/<task>/` inside the project — it is
gitignored. Never use /tmp or session scratchpads, and never `cd` — run every command
from the project root with repo-relative paths, so the project's pre-approved
permissions match and nothing prompts the user.

`<task>` is the dataset stem exactly as it appears in `data/` — `telco_churn`,
`german_credit`, `ames_housing`, `wine_quality`, `heart_disease` (for a user's own file,
its filename without extension and without a `_history`/`_test`/`_current` suffix).
Never invent a slug, never hyphenate it. Use ONLY these file names in the folder — reuse
(overwrite) them on later runs instead of inventing new ones:

| File | What it holds |
|---|---|
| `handoff.json` | Cached fit + prep notes (step 4) |
| `x_train.csv`, `y_train.csv` | Features / target uploaded for the fit |
| `x_test.csv`, `predictions.json` | Rows scored on request and TabPFN's downloaded output |
| `x_pasted.csv`, `predictions_pasted.json` | Pasted rows and their predictions (step 5) |
| `scores.csv` | Scored rows joined back to their features + prediction, for the user to open |
| `clean_history.csv`, `clean_test.csv` | Cleaned splits fed to `scripts/baseline.py` (comparisons only) |
| `baseline.json`, `evaluation.json` | Stdout of `scripts/baseline.py` / `scripts/evaluate.py` |
| `report.md`, `report-2.md`, … | Reports, only on request (see Reports) |

Never write per-task `.py` or `.sh` files, and never write metrics files other than the
two above.

## Data roles — what may be fit on

Datasets in `data/` come pre-split into role-named files (see `scripts/fetch_data.py`):

- `<name>_history.csv` — labeled rows. **The ONLY data a model may be fit on.** When the
  user points at their own labeled data ("my historical data", a file they name), that
  plays the history role.
- `<name>_test.csv` — the held-out rows WITH labels. **Evaluation only**: use it to
  compute metrics for a requested comparison. NEVER fit on it, and NEVER concatenate it
  (or any other split) with history to get "more training data" — that is leakage,
  full stop.
- `<name>_current.csv` — the SAME held-out rows with the target removed (same order).
  This is what gets scored — when the user asks. Because current == test minus labels,
  scored rows were never in history, and predictions on current can later be evaluated
  against test row-by-row.
There is no full-dataset copy in `data/` — the three role files are the whole dataset.
If a user drops in a file that looks like a combined copy of existing role files, don't
fit on it: it contains the test and current rows.

If it's ambiguous which file plays which role, say your reading in one line and proceed
with the conservative choice (fit on the smallest clearly-labeled history candidate) —
never resolve ambiguity by merging files.

## Deliverable rules

- **Do only what was asked.** Fit and predict are separately authorized: "train/fit/
  prepare a model on X" means fit on X, cache it, report it's ready to score — and STOP.
  Do not go hunting for a file to score. Predictions run only when the user asks for
  predictions, and only on the data they point at (or `_current` when they clearly mean
  the demo data). If they ask for predictions without saying on what, ask — one line.
- **The deliverable for a prediction ask is the chat answer**: predictions in the user's
  terms with calibrated probabilities and what they mean, plus a 2–3 line trace note
  (what you cleaned, what you delegated, wall-clock). Nothing else.
- **Baseline comparison only on request.** Run the XGBoost baseline only when the user
  asks how TabPFN compares / for metrics against another model — including when they ask
  AFTER predictions are done (the fit is cached; only the baseline needs running, plus a
  TabPFN predict on the same held-out split).
- **Reports only on request.** Never write a report file unprompted — see Reports below.

## Workflow

### 1. Frame the task
Restate the question as a prediction task: target column, task type
(binary/multiclass/regression), unit of prediction. For comparisons also pick
the metric that matches the question (imbalanced binary → ROC AUC / PR AUC, never
accuracy; asymmetric costs → say so).

### 2. Do the data work (inline, minimal)
Run ephemeral pandas (`uv run python - <<'EOF' ... EOF`) — never create per-task script
files. TabPFN handles categoricals, missing values, and scaling natively, so do NOT
build preprocessing pipelines. Fix only what is factually wrong or leaky:
- coerce numeric-as-string columns; parse dates.
- missing values: quantify; impute only when you know the true value (else leave them —
  TabPFN handles missing natively). State any choice you make.
- leakage: drop IDs, post-outcome columns, anything unknown at prediction time.
- split: the shipped `_history` / `_test` split already exists — use it for comparisons
  instead of re-splitting. Split by hand only for user-provided data that has no roles
  yet: stratified for iid data, time-based for anything temporal. For plain predictions,
  fit on all of history — but never on test/current rows (see Data roles).
- class balance: report it if it changes how the user should read the probabilities.
Keep a short list of the transformations you applied — it goes in `handoff.json` so
new rows get the same treatment later.

### 3. Baseline & evaluation — only when a comparison was requested
`uv run python scripts/baseline.py reports/<task>/clean_history.csv
reports/<task>/clean_test.csv <target> [--regression] > reports/<task>/baseline.json` —
a timed XGBoost baseline (native categorical support, defaults, no tuning). This is the
"what you'd get without TabPFN" half of the comparison. Feed it the same cleaned frames
TabPFN got (history and test with the step-2 fixes applied, ID columns dropped), never
the raw files — otherwise the two models see different data. (As a safety net the
script parses text columns whose values are all numbers as numeric; it reports them
under `parsed_as_numeric`.)

**All metrics come from `scripts/evaluate.py`.** Score TabPFN's downloaded predictions
on the test split with `uv run python scripts/evaluate.py data/<task>_test.csv <target>
reports/<task>/predictions.json [--regression] > reports/<task>/evaluation.json` (if a
step-2 fix changed the target, e.g. binarizing it, pass `clean_test.csv` instead).
Because `_current` is `_test` minus labels in the same order, predictions already made
on `_current` ARE test-split predictions — reuse them instead of predicting again.
The script reads the probability-column order from the `classes` in the
`handoff.json` next to the predictions file, so record `classes` before evaluating.
Report ONLY what the scripts print — ROC AUC for
classification, R2 for regression — plus wall-clock. Never compute extra metrics ad hoc
(no accuracy, PR AUC, Brier, log loss, MAE, calibration tables); if the user explicitly
asks for a specific additional metric, compute that one metric and say it's off the
standard list.

### 4. Hand off to TabPFN
The `tabpfn` MCP server chains ids:
files -> `train_set_upload_id` -> `fitted_train_set_id` -> `test_set_upload_id` -> prediction.

Write the splits as headered CSVs: `x_train.csv` and `x_test.csv` (identical columns, no
target), `y_train.csv` (exactly the one target column, rows aligned with x_train). Then:

1. `prepare_train_set_upload` (optionally `description`) → `train_set_upload_id` plus
   an `upload_command` (a curl command with a signed URL) for x_train and y_train.
2. Run each `upload_command` from the shell, replacing `<path/to/...>` with the
   repo-relative file (`reports/<task>/x_train.csv`, `reports/<task>/y_train.csv`).
   Upload is a plain HTTP PUT you perform yourself; there is no upload tool. Only ever
   send files from `reports/<task>/` to the signed URLs the server just returned.
3. `fit` with `train_set_upload_id` and `task` (`classification` | `regression`) →
   `fitted_train_set_id` immediately.
4. Poll `get_fit_status`, waiting `retry_in_secs` between polls, until completed.
5. `prepare_test_set_upload` with `fitted_train_set_id` → `test_set_upload_id` +
   `upload_command` for x_test; run it with `reports/<task>/x_test.csv` as above.
6. `predict` with `test_set_upload_id`, `fitted_train_set_id`, and the SAME `task` as
   the fit. Blocks; returns a signed download URL for a JSON array, one entry per test
   row (classification: per-class probabilities; regression: mean predictions).
   Download it with `curl --fail-with-body -o reports/<task>/predictions.json
   '<prediction_uri>'` and work with the file — don't read the whole array into
   context.

**Cache the fit.** After a successful fit, write `reports/<task>/handoff.json`:
`{fitted_train_set_id, train_set_upload_id, task, target, fit_file (the history file
fit on), feature_columns (ordered), prep_notes (the transformations applied, as short
strings), created}`. After the first predict, add `classes` from the predict
metadata — that is the column order of the probability arrays. A fit can serve
unlimited predict calls.

- Auth problems → tell the user to run `/mcp` and authenticate (keys at ux.priorlabs.ai).
- If the handoff fails, surface the error; never fall back to training a local model.

### 5. Scoring new data (pasted rows, a CSV snippet, a file)
When the user provides new rows for an existing task:
1. Parse whatever they pasted (with or without header; infer column mapping from names
   or position and confirm your reading back to them in one line).
2. Apply the same transformations listed in the cached `prep_notes`, and align to
   `feature_columns` exactly — same names, same order. Missing features: leave empty
   (TabPFN handles missing values natively); tell the user which were missing.
3. Write the aligned rows to `reports/<task>/x_pasted.csv`, reuse the cached
   `fitted_train_set_id`: `prepare_test_set_upload` → PUT → `predict`, download to
   `predictions_pasted.json`. No re-fit, no baseline — just steps 5–6 of the chain.
4. If the server rejects the cached id (expired), re-fit from the cached
   `train_set_upload_id` — or re-run the full chain — without bothering the user.
5. Answer in their terms: per-row prediction with probability, one line each.

## Reports — only when the user asks

Reports are living markdown documents in `reports/<task>/`, created and grown on
request ("write a report", "save this as a report", "add that to the report", "report
on these predictions"). File names are fixed — never make one up:

- **Create.** The first report for a task is always `reports/<task>/report.md`.
- **Add to it.** When the user asks to add something to "the report" — more predictions,
  a comparison they requested later, new caveats — edit the highest-numbered report in
  the folder (`report.md` if it's the only one) in place: extend or fill the relevant
  section, bump the Updated date. Do not regenerate the file or discard sections the
  user has seen.
- **New report.** If the user asks for a *new* report, create the next number:
  `report-2.md`, then `report-3.md`, and so on. Put what it's about in the title line,
  never in the file name. Leave the older reports untouched.
- **Benchmark added later.** "Add a baseline/comparison/benchmark to the report" means:
  run step 3 now on the shipped split, score TabPFN on the same split via the cached fit
  (or reuse `predictions.json` if `_current` was already scored), then add the
  Benchmark results section to the existing report.

**Sections.** Every report has the same headings in the same order. `Prediction
results` and `Test results` are always present; `Benchmark results` appears only once
a comparison has been asked for.

- **Prediction results** — what the user asked to have scored.
- **Test results** — how those predictions hold up against held-out labels, computed by
  `scripts/evaluate.py` only. Available when the scored rows are `_current` rows (their
  labels are in `_test`, same order) or pasted rows copied from `_current` (match them
  to their `_test` line). If the scored rows have no labels, write the single line
  "No held-out labels for these rows." — do not drop the heading.
- **Benchmark results** — XGBoost (`scripts/baseline.py`) vs TabPFN on the same held-out
  split.

**Template:**

```markdown
# <Dataset in plain words>: <what was predicted>
_Task: <task> · Report: <file name> · Created <YYYY-MM-DD> · Updated <YYYY-MM-DD>_

## Question
The user's question(s), near-verbatim, and how each was framed as a prediction task
(target, task type, unit of prediction).

## Data
Dataset and size, fit file (always `data/<task>_history.csv` for shipped data), scored
file, transformations applied (from prep_notes).

## Prediction results
The answer in the user's terms — prediction table(s) with probabilities or values
(top rows inline, full list in `scores.csv`), and how to read them (base rate, what the
probability means for the decision).

## Test results
The metric from `evaluation.json` (ROC AUC for classification, R2 for regression,
nothing else) on the scored rows vs their `_test` labels, n rows evaluated, and one
line on what it means. For single pasted rows: predicted vs actual label/value.

## Benchmark results
Only if requested:

| Model | <ROC AUC or R2> | Fit + predict wall-clock |
|---|---|---|
| XGBoost (defaults, no tuning) | from `baseline.json` | local CPU |
| TabPFN (hosted, no tuning) | from `evaluation.json` | incl. upload |

One paragraph on what the gap means for this user.

## Handoff trace
What the agent did (cleaning, framing, splits) vs what was delegated to TabPFN, with
wall-clock for each step.

## Notes
Caveats, data quirks, and decisions worth remembering.
```

## Tone
When comparing, narrate the refusal moment once, plainly: you won't hand-tune a model
because a tabular foundation model does that job better, with no tuning loop. Otherwise skip the
speech — just get the user their answer.
