"""Score TabPFN predictions against held-out labels, cookbook-style.

Mirrors the evaluation in Prior Labs' "TabPFN vs XGBoost" recipe
(docs.priorlabs.ai/cookbook/tabpfn_vs_xgboost): roc_auc_score on the positive-class
probability for classification (one-vs-rest macro average for multiclass), r2_score for
regression. No other metrics.

Usage: uv run python scripts/evaluate.py <truth.csv> <target> <predictions.json> [--regression]

<predictions.json> is the file downloaded from TabPFN predict: per-class probabilities
for classification, one number per row for regression. The probability columns follow
the `classes` list from the predict metadata; the script reads it from the `handoff.json`
next to the predictions file. Without it, it falls back to the labels' natural sort
order (numeric labels numerically, string labels alphabetically), which is how TabPFN
orders classes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score


def label(v) -> str:
    """Compare labels as text, so 3, 3.0 and "3" are the same class."""
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(v)


def class_order(pred_json: Path, y: pd.Series) -> tuple[list[str], str]:
    handoff = pred_json.parent / "handoff.json"
    if handoff.exists():
        classes = json.loads(handoff.read_text()).get("classes")
        if classes:
            return [label(c) for c in classes], str(handoff)
    return [label(c) for c in np.unique(y.dropna().to_numpy())], "natural sort order of the labels"


args = [a for a in sys.argv[1:] if a != "--regression"]
regression = "--regression" in sys.argv
truth_csv, target, pred_json = args

y_test = pd.read_csv(truth_csv)[target]
preds = np.asarray(json.load(open(pred_json)))
assert len(preds) == len(y_test), f"{len(preds)} predictions vs {len(y_test)} labels"

if regression:
    metrics = {"r2": float(r2_score(y_test, preds.reshape(-1)))}
else:
    classes, source = class_order(Path(pred_json), y_test)
    assert preds.shape[1] == len(classes), f"{preds.shape[1]} probability columns vs classes {classes}"
    y = np.array([label(v) for v in y_test])
    unknown = sorted(set(y) - set(classes))
    assert not unknown, f"test labels {unknown} are not among the predicted classes {classes}"
    if len(classes) == 2:
        metrics = {"roc_auc": float(roc_auc_score(y == classes[1], preds[:, 1]))}
    else:
        # One-vs-rest macro average over the classes that occur in the test labels.
        per_class = [roc_auc_score(y == c, preds[:, i]) for i, c in enumerate(classes) if c in set(y)]
        metrics = {"roc_auc_ovr": float(np.mean(per_class))}
    metrics["class_order_from"] = source

print(json.dumps({"n_test": len(y_test), "target": target, **metrics}, indent=2))
