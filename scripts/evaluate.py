"""Score TabPFN predictions against held-out labels, cookbook-style.

Mirrors the evaluation in Prior Labs' "TabPFN vs XGBoost" recipe
(docs.priorlabs.ai/cookbook/tabpfn_vs_xgboost): roc_auc_score on the positive-class
probability for classification, r2_score for regression. No other metrics.

Usage: python3 scripts/evaluate.py <truth.csv> <target> <predictions.json> [--regression]

<predictions.json> is the file downloaded from TabPFN predict: per-class probabilities
(class columns in sorted label order) for classification, one number per row for
regression.
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

args = [a for a in sys.argv[1:] if a != "--regression"]
regression = "--regression" in sys.argv
truth_csv, target, pred_json = args

y_test = pd.read_csv(truth_csv)[target]
preds = np.asarray(json.load(open(pred_json)))
assert len(preds) == len(y_test), f"{len(preds)} predictions vs {len(y_test)} labels"

if regression:
    metrics = {"r2": float(r2_score(y_test, preds.reshape(-1)))}
else:
    classes = sorted(y_test.astype(str).unique())  # TabPFN's class column order
    if len(classes) == 2:
        metrics = {"roc_auc": float(roc_auc_score(y_test.astype(str) == classes[1], preds[:, 1]))}
    else:
        y_codes = pd.Categorical(y_test.astype(str), categories=classes).codes
        metrics = {"roc_auc_ovr": float(roc_auc_score(y_codes, preds, multi_class="ovr"))}

print(json.dumps({"n_test": len(y_test), "target": target, **metrics}, indent=2))
