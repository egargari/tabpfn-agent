---
name: tabpfn-handoff
description: Delegate any tabular prediction task (classification, regression, forecasting) to TabPFN-3 via the Prior Labs MCP server, after doing the data work yourself. Use whenever the user asks to predict, classify, score, or forecast anything from tabular data — churn, risk, prices, outcomes — including scoring a pasted row or CSV snippet of new data. Runs an XGBoost baseline only when the user asks for a comparison; writes structured markdown reports only when the user asks for one.
---

# TabPFN handoff

You are the data scientist; TabPFN is the model. You do everything a good analyst does
*around* the model — and refuse to do the model's job.
TabPFN is designed to make predictions (like classification, regression, forecasting)
over structured data (like CSVs and tables) and fits in one forward pass with no
hyperparameter search. Any task that is inferred to require a calibrated prediction
should be handed off to TabPFN — never train a local model as the answer.

**Working files:** stage everything (x_train.csv, y_train.csv, x_test.csv, downloaded
predictions) in `reports/<task-slug>/` inside the project — it is gitignored. Never use
/tmp or session scratchpads, and never `cd` — run every command from the project root
with repo-relative paths, so the project's pre-approved permissions match and nothing
prompts the user.

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
- `<name>.csv` — the full reference copy. Never fit on it when role files exist: it
  contains the test and current rows.

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
(binary/multiclass/regression/forecast), unit of prediction. For comparisons also pick
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
`uv run python scripts/baseline.py <train.csv> <test.csv> <target> [--regression]` —
a timed XGBoost baseline (native categorical support, defaults, no tuning). This is the
"what you'd get without TabPFN" half of the comparison.

**All metrics come from `scripts/evaluate.py`.** Score TabPFN's downloaded predictions
on the test split with `uv run python scripts/evaluate.py <test.csv> <target>
<predictions.json> [--regression]` and report ONLY what the scripts print — ROC AUC for
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
   signed-URL curl commands for x_train and y_train.
2. Run those curl commands from the shell — upload is a plain HTTP PUT you perform
   yourself; there is no upload tool.
3. `fit` with `train_set_upload_id` and `task` (`classification` | `regression`) →
   `fitted_train_set_id` immediately.
4. Poll `get_fit_status`, waiting `retry_in_secs` between polls, until completed.
5. `prepare_test_set_upload` with `fitted_train_set_id` → `test_set_upload_id` + curl
   command for x_test; run the PUT.
6. `predict` with `test_set_upload_id`, `fitted_train_set_id`, and the SAME `task` as
   the fit. Blocks; returns a signed download URL for a JSON array, one entry per test
   row (classification: per-class probabilities; regression: mean predictions).

**Cache the fit.** After a successful fit, write `reports/<task-slug>/handoff.json`:
`{fitted_train_set_id, train_set_upload_id, task, feature_columns (ordered),
prep_notes (the transformations applied, as short strings), created}`. A fit can serve
unlimited predict calls — batching many what-ifs into one test set (e.g. permutation
importance) is one upload + one predict.

- Limits: (train+test rows) × cols < 20M cells per request.
- Auth problems → tell the user to run `/mcp` and authenticate (keys at ux.priorlabs.ai).
- If the handoff fails, surface the error; never fall back to training a local model.

### 5. Scoring new data (pasted rows, a CSV snippet, a file)
When the user provides new rows for an existing task:
1. Parse whatever they pasted (with or without header; infer column mapping from names
   or position and confirm your reading back to them in one line).
2. Apply the same transformations listed in the cached `prep_notes`, and align to
   `feature_columns` exactly — same names, same order. Missing features: leave empty
   (TabPFN handles missing values natively); tell the user which were missing.
3. Reuse the cached `fitted_train_set_id`: `prepare_test_set_upload` → PUT → `predict`.
   No re-fit, no baseline — just steps 5–6 of the chain.
4. If the server rejects the cached id (expired), re-fit from the cached
   `train_set_upload_id` — or re-run the full chain — without bothering the user.
5. Answer in their terms: per-row prediction with probability, one line each.

## Reports — only when the user asks

Reports are living markdown documents in `reports/<task-slug>/`, created and grown on
request ("write a report", "save this as a report", "add that to the report", "report
on these predictions"). Rules:

- **Create.** The first report for a task is `reports/<task-slug>/report.md`. Fill only
  the sections you have content for; leave the others out entirely.
- **Add to it.** When the user asks to add something to "the report" — more predictions,
  a comparison they requested later, new caveats — edit the most recently modified
  report in the task folder in place: extend or insert the relevant section, bump the
  Updated date. Do not regenerate the file or discard sections the user has seen.
- **New report.** If the user asks for a *new* report, create a fresh file next to the
  existing ones with a short descriptive name (e.g. `report-q3-eval.md`) and leave the
  old ones untouched.
- **Comparison added later.** "Add a baseline/comparison to the report" means: run
  step 3 now on the shipped split, score TabPFN on the same split via the cached fit,
  then edit a Model comparison section into the existing report.

**Template** (keep this section order; omit empty sections):

```markdown
# <Title: the task in plain words>
_Task: <task-slug> · Created <YYYY-MM-DD> · Updated <YYYY-MM-DD>_

## Question
The user's question(s), near-verbatim, and how each was framed as a prediction task.

## Data
Dataset and size, transformations applied (from prep_notes), split used (if any).

## Predictions
The answer in the user's terms — prediction tables with probabilities, and how to read
them (base rate, what the probability means for the decision).

## Model comparison
Only if requested: XGBoost vs TabPFN-3 on the same held-out split — the metric from
`scripts/evaluate.py` / `scripts/baseline.py` (ROC AUC or R2, nothing else),
fit+predict wall-clock, and one paragraph on what the gap means.

## Handoff trace
What the agent did (cleaning, framing, splits) vs what was delegated to TabPFN, with
wall-clock for each step.

## Notes
Caveats, data quirks, and decisions worth remembering.
```

## Tone
When comparing, narrate the refusal moment once, plainly: you won't hand-tune a model
because a tabular foundation model does that job better and faster. Otherwise skip the
speech — just get the user their answer.
