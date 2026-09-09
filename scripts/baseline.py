"""Timed XGBoost baseline for the TabPFN comparison.

Usage: python3 scripts/baseline.py <train.csv> <test.csv> <target> [--regression]

Only run when the user asked for a comparison — not for plain predictions.
Reports the metrics Prior Labs' own cookbook uses: ROC AUC for classification,
R2 for regression, plus fit+predict wall-clock. Nothing else.
"""

import json
import sys
import time

import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor


def to_xgb_frame(X: pd.DataFrame, categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Cast object columns to category dtype (XGBoost native categorical support).

    Test-set categories must reuse the train set's category levels so codes align.
    """
    X = X.copy()
    obj_cols = X.select_dtypes(include=["object", "string"]).columns
    if categories is None:
        categories = {c: pd.CategoricalDtype(X[c].astype("string").unique().dropna()) for c in obj_cols}
    for c, dtype in categories.items():
        vals = X[c].astype("string")
        X[c] = vals.where(vals.isin(dtype.categories)).astype(dtype)
    return X, categories


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--regression"]
    regression = "--regression" in sys.argv
    train_path, test_path, target = args
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    X_train, cats = to_xgb_frame(train.drop(columns=[target]))
    X_test, _ = to_xgb_frame(test.drop(columns=[target]), cats)
    y_train, y_test = train[target], test[target]

    if regression:
        model = XGBRegressor(tree_method="hist", enable_categorical=True, random_state=0)
    else:
        y_train, class_labels = pd.factorize(y_train)
        y_test = pd.Categorical(y_test, categories=class_labels).codes
        model = XGBClassifier(tree_method="hist", enable_categorical=True, random_state=0)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    if regression:
        pred = model.predict(X_test)
        elapsed = time.perf_counter() - start
        metrics = {"r2": float(r2_score(y_test, pred))}
    else:
        proba = model.predict_proba(X_test)
        elapsed = time.perf_counter() - start
        if proba.shape[1] == 2:
            metrics = {"roc_auc": float(roc_auc_score(y_test, proba[:, 1]))}
        else:
            metrics = {"roc_auc_ovr": float(roc_auc_score(y_test, proba, multi_class="ovr"))}

    report = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "target": target,
        "models": {"xgboost": {"fit_predict_seconds": round(elapsed, 3), **metrics}},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
