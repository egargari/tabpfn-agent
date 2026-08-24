"""Render a comparison receipt from a JSON payload — generic across tasks.

Usage: python3 scripts/report.py <payload.json>

Payload shape (all fields optional except task):
{
  "task": "telco-churn",                      # slug; receipt goes to reports/<task>/receipt.md
  "question": "the user's question, verbatim",
  "framing": "how it was framed as a prediction task, incl. caveats",
  "trace": [{"actor": "agent"|"tabpfn", "step": "...", "seconds": 1.4}],
  "scoreboard": [{"model": "XGBoost", "metrics": {"ROC AUC": 0.825}, "seconds": 0.21}],
  "notes": ["calibration note", "data-work note", ...],
  "answer": "markdown block with the actual answer"
}
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text())
    lines = [f"# Receipt: {payload['task']}", ""]
    lines.append(f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append("")

    if q := payload.get("question"):
        lines += ["## The question", "", f"> {q}", ""]
    if f := payload.get("framing"):
        lines += ["## Framing", "", f, ""]

    if trace := payload.get("trace"):
        lines += ["## Handoff trace", "", "| Who | Step | Time |", "|---|---|---|"]
        for t in trace:
            secs = f"{t['seconds']:.2f}s" if "seconds" in t else ""
            lines.append(f"| {t.get('actor', 'agent')} | {t['step']} | {secs} |")
        lines.append("")

    if scoreboard := payload.get("scoreboard"):
        metric_names = list(dict.fromkeys(k for row in scoreboard for k in row.get("metrics", {})))
        lines += ["## Scoreboard", "", "| Model | " + " | ".join(metric_names) + " | Fit+predict |"]
        lines.append("|---|" + "---|" * (len(metric_names) + 1))
        for row in scoreboard:
            cells = [fmt(row["metrics"].get(m, "—")) for m in metric_names]
            secs = f"{row['seconds']:.2f}s" if "seconds" in row else "—"
            lines.append(f"| {row['model']} | " + " | ".join(cells) + f" | {secs} |")
        lines.append("")

    if notes := payload.get("notes"):
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    if answer := payload.get("answer"):
        lines += ["## Answer", "", answer, ""]

    out = Path("reports") / payload["task"] / "receipt.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
