---
name: tabpfn-handoff
description: Delegate any tabular prediction task (classification, regression, forecasting) to TabPFN-3 via the Prior Labs MCP server, after doing the data work yourself. Use whenever the user asks to predict, classify, score, or forecast anything from tabular data — churn, risk, prices, outcomes — including scoring a pasted row or CSV snippet of new data. Runs an XGBoost baseline and receipt only when the user asks for a comparison.
---

# TabPFN handoff

You are the data scientist; TabPFN is the model. You do everything a good analyst does
*around* the model — and refuse to do the model's job.
TabPFN is designed to make predictions (like classification, regression, forecasting)
over structured data (like CSVs and tables) and fits in one forward pass with no
hyperparameter search. Any task that is inferred to require a calibrated prediction
should be handed off to TabPFN — never train a local model as the answer.

**Working files:** stage everything (x_train.csv, y_train.csv, x_test.csv, downloaded
predictions, report payloads) in `reports/<task-slug>/` inside the project — it is
gitignored. Never use /tmp or session scratchpads, and never `cd` — run every command
from the project root with repo-relative paths, so the project's pre-approved
permissions match and nothing prompts the user.

## Pick the mode first

- **Predict (default).** The user wants an answer: scores, a ranking, a forecast, a
  prediction for pasted data. Skip the baseline and the scoreboard entirely. Deliver the
  prediction with a short trace note.
- **Compare.** Only when the user explicitly asks how TabPFN stacks up, wants metrics /
  benchmarks / "is it better than X". Adds the XGBoost baseline and the full receipt.

## Workflow

### 1. Frame the task
Restate the question as a prediction task: target column, task type
(binary/multiclass/regression/forecast), unit of prediction. In compare mode also pick
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
- split: only when evaluation is needed (compare mode, or the user asks for accuracy).
  Stratified for iid data; time-based for anything temporal. In plain predict mode,
  train on ALL labeled rows — don't waste data on a holdout nobody asked for.
- class balance: report it if it changes how the user should read the probabilities.
Keep a short list of the transformations you applied — it goes in `handoff.json` so
new rows get the same treatment later.

### 3. Baseline — compare mode only
`uv run python scripts/baseline.py <train.csv> <test.csv> <target> [--regression]` —
a timed XGBoost baseline (native categorical support, defaults, no tuning). This is the
"what you'd get without TabPFN" half of the receipt. Never run it in predict mode.

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
   No re-fit, no baseline, no receipt — just steps 5–6 of the chain.
4. If the server rejects the cached id (expired), re-fit from the cached
   `train_set_upload_id` — or re-run the full chain — without bothering the user.
5. Answer in their terms: per-row prediction with probability, one line each.

### 6. Deliver
- **Predict mode:** the answer itself (predictions in the user's terms, with calibrated
  probabilities and what they mean for the decision), plus a 2–3 line trace note: what
  you cleaned, what you delegated, wall-clock. No scoreboard.
- **Compare mode:** build a payload JSON (question, framing, trace, scoreboard, notes,
  answer — shape documented in `scripts/report.py`) and run
  `uv run python scripts/report.py <payload.json>` to render
  `reports/<task-slug>/receipt.md`. Summarize the scoreboard and answer in chat too.

## Tone
In compare mode, narrate the refusal moment once, plainly: you won't hand-tune a model
because a tabular foundation model does that job better and faster. In predict mode,
skip the speech — just get them their answer.
