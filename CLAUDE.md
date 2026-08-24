# tabpfn-agent

This repo is a demo: an agent that does the data work itself but **never hand-rolls model
selection** — every fit/predict on tabular data is delegated to TabPFN-3 via the official
Prior Labs MCP server (configured in `.mcp.json`, OAuth via `/mcp`).

## Rules

- Run all Python through the uv-managed venv: `uv run python ...` (`uv sync` to set up).
- For ANY prediction task on tabular data, use the `tabpfn-handoff` skill. Default mode
  is PREDICT: just deliver the prediction — no baseline, no scoreboard. Only run the
  XGBoost baseline (`scripts/baseline.py`) and write a receipt when the user explicitly
  asks for a comparison or metrics. Never train a local model as the answer.
- Users may paste rows or CSV snippets of new data to score; reuse the cached fit in
  `reports/<task>/handoff.json` (see the skill, step 5).
- Use the hosted API/MCP path only. Do not download local TabPFN weights: the TabPFN-3
  weights license is non-commercial, and this demo must stay clean for public use.
- Datasets live in `data/` (see `scripts/fetch_data.py` to pull more). The headliner is
  `data/telco_churn.csv` (OpenML 42178): 7,043 customers, target `Churn`, ~26.5% positive.
  Known trap: `TotalCharges` is a string column with blank values for new customers.
  A ready-made stratified 80/20 split ships as `telco_churn_train.csv` /
  `telco_churn_test.csv` (random_state=0) — use it for compare mode instead of
  re-splitting; use the full file when training for plain predictions.
- Receipts (`reports/<task>/receipt.md`: XGBoost vs TabPFN scoreboard, handoff trace,
  calibration note) are written in compare mode only. In predict mode the deliverable is
  the answer plus a 2-3 line trace note.

## Facts (verified August 2026)

- TabPFN-3 is current. API limits: 1M train rows, 2,000 columns, 160 classes,
  (train+test rows) × cols < 20M cells per request, 100M cells/day.
- Thinking mode (TabPFN-3-Plus) and native text columns are API-only, separate quota.
- Docs index: https://docs.priorlabs.ai/llms.txt — fetch it rather than trusting memory.
