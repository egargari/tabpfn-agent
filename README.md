# Predictive Agent with TabPFN and Claude

A [Claude Code](https://claude.com/claude-code) project showing how an agent can
recognize a **prediction task** and hand it to a specialized model as a tool call,
instead of guessing with the LLM or training a model itself.

Ask a question about a table ("which customers will churn?", "what will this house
sell for?") and the agent does the analyst's work: framing the question, cleaning the
data, checking for leakage. It then delegates the fit and prediction to
**[TabPFN](https://priorlabs.ai)**, Prior Labs' tabular foundation model, through the
official TabPFN MCP server. Questions that aren't predictions ("how many customers
churned?") are answered straight from the data.

```
you ──ask──▶ Claude (frame, clean, check leakage)
                 │
                 └──tool call──▶ TabPFN MCP server ──▶ TabPFN (fit + predict)
                 ◀──calibrated probabilities──┘
```

## Setup

You need:

- [Claude Code](https://code.claude.com/docs/en/setup) (needs a paid Claude plan or an
  Anthropic Console account)
- Python, either way:
  - **[uv](https://docs.astral.sh/uv/)** (recommended): installs the exact package
    versions the demo was recorded with, and fetches a suitable Python if needed.
  - **or Python 3.9+ with pip**: works the same; package versions may differ slightly,
    so benchmark numbers can vary a little.
- A free [Prior Labs account](https://ux.priorlabs.ai)

```bash
git clone <this-repo>
cd tabpfn-agent
```

Then install the dependencies with **one** of these:

```bash
uv sync
```

```bash
pip install -r requirements.txt
```

Start Claude Code with `claude`, run `/mcp` once, and sign in to the `tabpfn` server
with your Prior Labs account (opens your browser).

## Try it

The repo ships with a telco customer-churn dataset: 5,634 customers with known
outcomes to learn from, plus 1,409 "current" customers to score. These questions, in
roughly this order, make a complete demo:

1. *"How many customers do we have, and what share of them churned?"*: not a
   prediction, so the agent answers from the data with no TabPFN call.
2. *"Fit a churn model on our customer history."*: the agent cleans the data (it
   catches the `TotalCharges` column stored as text), fits TabPFN, and stops.
3. *"Which of our current customers are most likely to churn? Give me the top 10."*
4. *"Is this customer likely to churn?"* followed by a pasted row, e.g. the last line of
   `data/telco_churn_current.csv` (the true answer is `Yes`):
   `Female,0,Yes,Yes,1,Yes,Yes,'Fiber optic',Yes,No,Yes,No,Yes,Yes,Month-to-month,Yes,'Electronic check',102.45,102.45`
5. *"What about these three?"* with a few rows pasted as CSV. Pasted rows reuse the
   cached fit, so there's no retraining.
6. *"How well did those predictions do against the real outcomes?"*
7. *"How does that compare to XGBoost?"*: runs an untuned XGBoost baseline on the same
   rows and compares ROC AUC.
8. *"Write a report on this."*, then *"Add the XGBoost comparison to the report."*

Comparisons and reports only happen when you ask. Reports are saved to
`reports/<dataset>/report.md` (`report-2.md` and so on for new ones) with the same
sections every time: Question, Data, Prediction results, Test results, Benchmark
results (once you've asked for a comparison), Handoff trace, Notes.

## Datasets

| Dataset | Question | Target | Rows |
|---|---|---|---|
| `telco_churn` (included) | Will this customer churn? | `Churn` | 7,043 |
| `ames_housing` (included) | What will this house sell for? | `SalePrice` | 1,460 |
| `german_credit` | Will this loan applicant default? | `class` | 1,000 |
| `wine_quality` | What quality score (3–9) will this wine get? | `quality` | 6,497 |
| `heart_disease` | Is heart disease present (grade 0–4)? | `num` | 303 |

Download the other three (no credentials needed). With uv:

```bash
uv sync --extra datasets
```

```bash
uv run python scripts/fetch_data.py german_credit wine_quality heart_disease
```

Or with pip:

```bash
pip install -r requirements-datasets.txt
```

```bash
python scripts/fetch_data.py german_credit wine_quality heart_disease
```

Each dataset is split into three files in `data/`:

- `_history.csv`: labeled rows (80%), the only data models learn from.
- `_current.csv`: the other 20%, without labels. This is what you ask to score.
- `_test.csv`: the same 20% with labels, so predictions can be checked afterwards.

You can also drop your own CSV into `data/` and ask about it.

## Permissions

The shared [`.claude/settings.json`](.claude/settings.json) pre-approves what the demo
needs: the TabPFN MCP tools, the project's Python (`uv run`), and read-only file
commands.

It does **not** pre-approve `curl`, which the agent uses to upload your CSVs to Prior
Labs and download predictions (two or three commands per prediction). A blanket curl
rule would let the agent send any file anywhere without asking. So Claude Code asks
first: check that the URL is Prior Labs storage
(`storage.googleapis.com/p2or-datasets-prod/...`) and the file is in `reports/`. To
skip these prompts on your own machine, add this to your personal, gitignored
`.claude/settings.local.json`:

```json
{ "permissions": { "allow": ["Bash(curl:*)"] } }
```

## Repo layout

| Path | What it is |
|---|---|
| `.claude/skills/tabpfn-handoff/SKILL.md` | The skill: when to hand off, the data work, the MCP steps, the report template |
| `CLAUDE.md` | Project rules the agent always follows |
| `.mcp.json` | The Prior Labs MCP server (`api.priorlabs.ai/mcp/server`) |
| `.claude/settings.json` | Pre-approved permissions |
| `data/` | The datasets, split into history / current / test |
| `scripts/fetch_data.py` | Downloads the datasets and writes the splits |
| `scripts/baseline.py` | XGBoost baseline, only for comparisons |
| `scripts/evaluate.py` | Scores predictions against the held-out labels (ROC AUC / R²) |
| `reports/<dataset>/` | Created as you go: the cached fit, working files, reports (gitignored) |

## Notes

- Predictions run on Prior Labs' hosted API, so your data is uploaded to it. The free
  account has a daily allowance.
- On macOS, XGBoost needs the OpenMP runtime: `brew install libomp`.
- The local TabPFN weights aren't used: the hosted API is the point of the demo, and the
  local-weights license is non-commercial.
