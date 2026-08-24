# Predictive Agent with TabPFN and Claude

This is a [Claude Code](https://claude.com/claude-code) project where the agent does the
data work — loading, cleaning, leakage checks — and hands over to
**[TabPFN](https://priorlabs.ai)**, Prior Labs' tabular foundation model, through the
official TabPFN MCP server, when the user request asks for a prediction (classification, regression, forecasting etc) over a structured dataset (like a CSV). This way, instead of using an LLM for a task that should be explainable and deterministic, the agent knows to hand-off to a specialized tabular foundation model. 

## Setup

You need:

- [Claude Code](https://claude.com/claude-code)
- [uv](https://docs.astral.sh/uv/)
- A free [Prior Labs account](https://ux.priorlabs.ai)

```bash
git clone <this-repo>
cd tabpfn-agent
uv sync
claude
```

Inside Claude Code, run `/mcp` once and authenticate the `tabpfn` server with your Prior
Labs account (OAuth, opens your browser). That's the only permission step — the project
settings pre-approve everything else the demo needs.

## Using it

**Ask for a prediction.** The repo ships with a telco customer-churn dataset
(`data/telco_churn.csv`, 7,043 customers). Just ask:

> Which of my customers are most likely to churn, and what's driving it?

The agent inspects and cleans the data (it will catch the string-typed `TotalCharges`
column on its own), delegates the fit to TabPFN-3, and answers with a ranked list of
at-risk customers, calibrated probabilities, and the drivers behind them.

**Score new data.** Paste a row — or a few — straight into the chat:

> What's the churn risk for this customer?
> Female, senior, no partner, tenure 2, fiber optic, month-to-month, electronic check, monthly charges 85.70

The agent maps your row onto the trained features, reuses the cached TabPFN fit (no
retraining), and returns the probability. Missing fields are fine — TabPFN handles
missing values natively.

**Ask for a comparison.** Benchmarks only happen when you ask:

> How does TabPFN do on this compared to XGBoost?

This runs a timed, untuned XGBoost baseline on a held-out split, hands the same split to
TabPFN, and writes a full receipt to `reports/<task>/receipt.md`: a timestamped trace of
what the agent did vs. what it delegated, the scoreboard, and a calibration note.

**Bring your own data.** Drop any CSV into `data/` and ask a question about it — the
skill works for any tabular classification or regression task. Or pull one of the
prepared demo datasets (no credentials needed):

```bash
uv run python scripts/fetch_data.py german_credit ames_housing wine_quality
```

Available: `telco`, `german_credit`, `ames_housing`, `wine_quality`, `heart_disease`,
`premier_league`.

## How it works

The behavior lives in one skill,
[`.claude/skills/tabpfn-handoff/SKILL.md`](.claude/skills/tabpfn-handoff/SKILL.md).
Whenever a prediction is asked for, it has the agent:

1. Frame the question as a prediction task (target, task type, the right metric).
2. Do the minimal data work inline — type coercion, leakage checks. TabPFN handles
   categoricals and missing values natively, so there's no preprocessing pipeline.
3. Hand off to the TabPFN MCP server: upload the feature/target CSVs via signed URLs,
   `fit`, then `predict`. The fit and the cleaning decisions are cached in
   `reports/<task>/handoff.json` so follow-up questions and pasted rows reuse them.
4. Answer in your terms — never by training a local model itself.

```
you ──ask──▶ Claude (data work: clean, frame, split)
                 │
                 └──handoff──▶ TabPFN MCP server ──▶ TabPFN-3 (fit + predict, one forward pass)
                 ◀──calibrated probabilities──┘
```

## Repo layout

| Path | What it is |
|---|---|
| `.claude/skills/tabpfn-handoff/` | The skill: when to hand off, the data-work checklist, the MCP tool chain |
| `.claude/settings.json` | Pre-approved permissions so the demo runs without prompts |
| `.mcp.json` | The official Prior Labs MCP server (`api.priorlabs.ai/mcp/server`) |
| `CLAUDE.md` | Project rules the agent follows |
| `data/telco_churn.csv` | Headline dataset (OpenML 42178), full 7,043 rows |
| `data/telco_churn_train.csv` / `_test.csv` | Ready-made stratified 80/20 split of the same data |
| `scripts/baseline.py` | Timed XGBoost baseline, used only in comparison mode |
| `scripts/report.py` | Renders the comparison receipt from a JSON payload, any task |
| `scripts/fetch_data.py` | Downloads additional credential-free demo datasets |
| `reports/` | Generated per-task: predictions, cached fits, comparison receipts (gitignored) |

## Notes

- Predictions run on Prior Labs' hosted API — your data is uploaded to it. Limits per
  request: (train rows + test rows) × columns < 20M cells; free daily allowance applies.
- Comparison mode needs XGBoost; on macOS install the OpenMP runtime first:
  `brew install libomp`.
- The local TabPFN weights are not used anywhere: the hosted API is the point of the
  demo (and the local-weights license is non-commercial).
