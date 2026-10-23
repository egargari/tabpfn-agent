# Predictive Agent with TabPFN and Claude

This is a [Claude Code](https://claude.com/claude-code) project where the agent does the
data work — loading, cleaning, leakage checks — and hands over to
**[TabPFN](https://priorlabs.ai)**, Prior Labs' tabular foundation model, through the
official TabPFN MCP server, whenever a request turns out to be a prediction task
(classification or regression) over a structured dataset like a CSV. Instead of
guessing with the LLM or hand-rolling a model, the agent recognizes the task and makes a
tool call to a specialized tabular foundation model.

## Setup

You need:

- [Claude Code](https://claude.com/claude-code)
- [uv](https://docs.astral.sh/uv/) — or just Python 3.10+ with pip (see below)
- A free [Prior Labs account](https://ux.priorlabs.ai)

```bash
git clone <this-repo>
cd tabpfn-agent
uv sync
claude
```

**No uv? Use pip.** In a sandbox or environment that only has pip, install the same
dependencies from the requirements file instead of `uv sync`:

```bash
pip install -r requirements.txt
```

Use `requirements-datasets.txt` instead if you also want the UCI datasets (wine quality,
heart disease). The agent notices when uv is missing and runs the same commands with
plain `python`; everywhere this README says `uv run python`, `python` works too.

Inside Claude Code, run `/mcp` once and authenticate the `tabpfn` server with your Prior
Labs account (OAuth, opens your browser). The project settings pre-approve the commands
the demo needs, so you shouldn't get permission prompts in the normal flow — see
[Permissions](#permissions) for what's pre-approved and why.

## Using it

**Ask for a prediction.** The repo ships with a telco customer-churn dataset
(7,043 customers, pre-split into history and held-out current customers). Just ask:

> Which of my current customers are most likely to churn?

The agent inspects and cleans the data (it will catch the string-typed `TotalCharges`
column on its own), delegates the fit to TabPFN, and answers with a ranked list of
at-risk customers and their calibrated churn probabilities.

**Score new data.** Paste a row — or a few — straight into the chat:

> What's the churn risk for this customer?
> Female, senior, no partner, tenure 2, fiber optic, month-to-month, electronic check, monthly charges 85.70

Or copy a real row from `data/telco_churn_current.csv` — its true label is on the same
line of `data/telco_churn_test.csv`, so you can check the answer afterwards.

The agent maps your row onto the trained features, reuses the cached TabPFN fit (no
retraining), and returns the probability. Missing fields are fine — TabPFN handles
missing values natively.

**Check it against held-out labels.** The current customers are the held-out split, so
their true labels exist in `data/telco_churn_test.csv`:

> How well did those predictions do?

**Ask for a comparison.** Benchmarks only happen when you ask:

> How does TabPFN do on this compared to XGBoost?

This runs a timed, untuned XGBoost baseline on the shipped held-out split, scores TabPFN
on the same split, and answers with ROC AUC (classification) or R² (regression) for
both. You can ask at any point — even after your predictions are done.

**Ask for a report.** Reports are also opt-in, and they're living documents:

> Write a report on those predictions.
> Add the XGBoost comparison to the report.
> Start a new report for this quarter's eval.

The agent maintains markdown files in `reports/<task>/`, where `<task>` is the dataset
name (e.g. `reports/telco_churn/`). The first ask creates `report.md`, "add to the
report" edits it in place, and "new report" creates `report-2.md`, `report-3.md`, and
so on. Every report has the same sections: Question, Data, **Prediction results**,
**Test results** (predictions scored against the held-out labels), **Benchmark
results** (XGBoost vs TabPFN — only once you've asked for a comparison), Handoff trace,
and Notes. The template lives in the skill.

### Questions to try on telco churn

In roughly this order, they make a complete demo:

1. *"How many customers do we have, and what share of them churned?"* — not a
   prediction, so the agent answers from the data directly, with no TabPFN call.
2. *"Fit a churn model on our customer history."* — fit only: the agent cleans the
   data, fits TabPFN, caches the fit, and stops. It doesn't score anything you didn't
   ask for.
3. *"Which of our current customers are most likely to churn? Give me the top 10."*
4. *"Is this customer likely to churn?"* followed by a pasted row, e.g. the last line of
   `data/telco_churn_current.csv`:
   `Female,0,Yes,Yes,1,Yes,Yes,'Fiber optic',Yes,No,Yes,No,Yes,Yes,Month-to-month,Yes,'Electronic check',102.45,102.45`
   (the true label is `Yes`).
5. *"What about these three?"* with a few rows pasted as CSV — one upload, one predict,
   same cached fit.
6. *"How many of the current customers would you flag at a 50% threshold?"*
7. *"How well did those predictions do against the real outcomes?"* — Test results.
8. *"How does that compare to XGBoost?"* — Benchmark results.
9. *"Write a report on this."*, then *"Add the XGBoost comparison to the report."*

**More datasets.** Telco churn and Ames housing ship with the repo. Pull the others
(no credentials needed):

```bash
uv sync --extra datasets
```

(with pip: `pip install -r requirements-datasets.txt`)

```bash
uv run python scripts/fetch_data.py german_credit wine_quality heart_disease
```

| Dataset | Task | Target | Rows |
|---|---|---|---|
| `telco_churn` (shipped) | Binary classification: will this customer churn? | `Churn` | 7,043 |
| `ames_housing` (shipped) | Regression: what will this house sell for? | `SalePrice` | 1,460 |
| `german_credit` | Binary classification: will this loan applicant default? | `class` | 1,000 |
| `wine_quality` | Multiclass or regression: quality score 3–9 | `quality` | 6,497 |
| `heart_disease` | Multiclass (0–4) or binary: heart disease present? | `num` | 303 |

Every dataset arrives pre-split into roles: `_history.csv` (labeled — the only thing
models are fit on) and a 20% holdout that exists twice: `_test.csv` with labels (for
evaluation) and `_current.csv` without them (what you ask to score). They're the same
rows in the same order, so anything scored was never trained on and can be checked
against test afterwards.

**Bring your own data.** Drop any CSV into `data/` and ask a question about it — the
skill works for any tabular classification or regression task. If it has no role
split, the agent makes one before it evaluates anything.

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

Questions that aren't predictions ("how many customers churned?", "what's the average
bill?") are answered directly from the data, with no handoff.

```
you ──ask──▶ Claude (data work: clean, frame, split)
                 │
                 └──handoff──▶ TabPFN MCP server ──▶ TabPFN (fit + predict, one forward pass)
                 ◀──calibrated probabilities──┘
```

## Permissions

Claude Code asks before running any shell command that isn't pre-approved. The shared
project settings in [`.claude/settings.json`](.claude/settings.json) pre-approve only
what the demo needs:

| Rule | Why |
|---|---|
| `mcp__tabpfn` | The TabPFN MCP tools (fit, predict, upload preparation) |
| `Bash(uv run:*)`, `Bash(uv sync:*)` | The project's Python: inline pandas, `scripts/*.py` |
| `Bash(python scripts/:*)`, `Bash(python -:*)`, `Bash(pip install -r requirements:*)` | The same, for pip-only environments without uv |
| `Bash(ls:*)`, `head`, `wc`, `cat`, `mkdir`, `sleep`, `date` | Read-only file inspection and polling |

**Why curl isn't on the list — and the prompts you'll see.** The TabPFN MCP server
hands back signed URLs, and the agent uploads your CSVs and downloads predictions with
`curl`: two or three commands per prediction. These go only to Prior Labs' storage
(`https://storage.googleapis.com/p2or-datasets-prod/...`). The shared settings
deliberately don't pre-approve curl, because `Bash(curl:*)` would let the agent send any
file to any host without asking, for everyone who clones the repo. So Claude Code asks
you before each transfer: check the URL is Prior Labs storage and the file is in
`reports/`, then approve.

To stop the prompts on your own machine, add the rule to your personal, gitignored
`.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": ["Bash(curl:*)"]
  }
}
```

That's your call to make for yourself; it isn't shipped as a default. (A curl rule
restricted to the Prior Labs URL won't help much: argument-matching rules are easy to
get around, see the Claude Code [permissions docs](https://code.claude.com/docs/en/permissions).)

If you'd rather not approve anything from this repo automatically, delete the `allow`
list and approve commands one by one. Choosing "Yes, and don't ask again" saves a rule
in your own `.claude/settings.local.json`. Note that `Bash(uv run:*)` and
`Bash(python -:*)` are the broad ones: they let the agent run any Python, which is what
the inline data work relies on.

## Repo layout

| Path | What it is |
|---|---|
| `.claude/skills/tabpfn-handoff/` | The skill: when to hand off, the data-work checklist, the MCP tool chain |
| `.claude/settings.json` | Pre-approved permissions so the demo runs without prompts (see [Permissions](#permissions)) |
| `.mcp.json` | The official Prior Labs MCP server (`api.priorlabs.ai/mcp/server`) |
| `CLAUDE.md` | Project rules the agent follows |
| `data/<name>_history.csv` | Labeled rows (80%) — the only data models are fit on |
| `data/<name>_test.csv` | Held-out rows (20%) with labels — evaluation only |
| `data/<name>_current.csv` | The same 20% with labels removed — the rows you ask to score |
| `pyproject.toml`, `uv.lock` | Dependencies for `uv sync` |
| `requirements.txt`, `requirements-datasets.txt` | The same dependencies for `pip install -r` (no uv needed) |
| `scripts/baseline.py` | Timed XGBoost baseline, run only when a comparison is requested |
| `scripts/evaluate.py` | Scores any predictions against held-out labels — the single source of metrics (ROC AUC / R²) |
| `scripts/fetch_data.py` | Downloads the demo datasets and writes the role splits |
| `reports/<task>/` | Generated per dataset: `handoff.json` (cached fit), working files, and on-request `report.md` (gitignored) |

## Notes

- Predictions run on Prior Labs' hosted API — your data is uploaded to it. Limits per
  request: (train rows + test rows) × columns < 20M cells; free daily allowance applies.
- Comparison mode needs XGBoost; on macOS install the OpenMP runtime first:
  `brew install libomp`.
- The local TabPFN weights are not used anywhere: the hosted API is the point of the
  demo (and the local-weights license is non-commercial).
