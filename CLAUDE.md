# tabpfn-agent

This repo is a demo: an agent that does the data work itself but **never hand-rolls model
selection** — every fit/predict on tabular data is delegated to TabPFN via the official
Prior Labs MCP server (configured in `.mcp.json`, OAuth via `/mcp`).

## Rules

- Run all Python through the uv-managed venv: `uv run python ...` (`uv sync` to set up).
- For ANY prediction task on tabular data, use the `tabpfn-handoff` skill. Default mode
  is PREDICT: just deliver the prediction — no baseline, no scoreboard. Only run the
  XGBoost baseline (`scripts/baseline.py`) when the user explicitly asks for a
  comparison or metrics. Never train a local model as the answer.
- Users may paste rows or CSV snippets of new data to score; reuse the cached fit in
  `reports/<task>/handoff.json` (see the skill, step 5).
- Move files to/from TabPFN with `scripts/transfer.py` (upload/download), never raw
  curl: it only talks to Prior Labs storage and only touches `reports/`, which is why it
  can run without permission prompts.
- Use the hosted API/MCP path only. Do not download local TabPFN weights: the TabPFN
  weights license is non-commercial, and this demo must stay clean for public use.
- Datasets live in `data/` as role-named splits (created by `scripts/fetch_data.py`,
  80/20, random_state=0): `<name>_history.csv` = labeled rows, the ONLY data to fit on;
  `<name>_test.csv` = the held-out 20% with labels, evaluation only — never fit on it,
  never concatenate splits; `<name>_current.csv` = the SAME held-out rows minus the
  target, what gets scored on request (evaluable row-by-row against test). There is no
  full-dataset copy in `data/`, by design.
- Headliner: telco churn (OpenML 42178), target `Churn`, ~26.5% positive. Known trap:
  `TotalCharges` is a string column with blank values for new (tenure-0) customers.
- Fit and predict are separately authorized: a request to fit/train a model ends after
  the fit is cached and reported ready. Never pick data to score on your own.
  `data/telco_churn_current.csv` is the test split with the `Churn` column removed (same
  row order, same quoting) — learners copy rows from it to paste as "new" customers to
  score.
- `<task>` is always the dataset stem from `data/` (e.g. `telco_churn`), and working files
  in `reports/<task>/` use the fixed names listed in the skill — never invent names.
- Reports are written ONLY when the user asks for one, and are living markdown files:
  `reports/<task>/report.md` first, `report-2.md`, `report-3.md`, … on "new report";
  "add to the report" edits the highest-numbered one. Every report has `Prediction
  results` and `Test results` sections; `Benchmark results` (XGBoost vs TabPFN) is added
  only when a comparison is asked for. Template in the skill. Default deliverable is
  always the chat answer plus a 2-3 line trace note.

## Facts (verified September 2026)

- The hosted MCP server picks the model version; nothing to update locally. Call it
  "TabPFN" in answers and reports — don't name a version number.
- API limits: 1M train rows, 2,000 columns, 160 classes, (train+test rows) × cols < 20M
  cells per request, 100M cells/day.
- Thinking mode and native text columns are API-only, separate quota.
- Docs index: https://docs.priorlabs.ai/llms.txt — fetch it rather than trusting memory.
