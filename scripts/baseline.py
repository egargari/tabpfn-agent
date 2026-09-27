"""Timed XGBoost baseline for the TabPFN comparison.

Usage: uv run python scripts/baseline.py <train.csv> <test.csv> <target> [--regression]

Only run when the user asked for a comparison — not for plain predictions.
Pass the same cleaned frames TabPFN was fit on. As a safety net for raw files, text
columns whose non-blank values are all numbers (e.g. numbers exported as strings, possibly
quoted, with blanks for missing) are parsed as numeric, so XGBoost doesn't treat them as categories.
Reports the metrics Prior Labs' own cookbook uses: ROC AUC for classification,
R2 for regression, plus fit+predict wall-clock. Nothing else.
"""

import json
import sys
import time

import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor


def unquote(col: pd.Series) -> pd.Series:
    """Strip whitespace and wrapping quotes, so ' 12.5 ', "12.5" and ' ' read as 12.5 / blank."""
    return col.astype("string").str.strip().str.strip("'\"").str.strip()


def numeric_text_columns(X: pd.DataFrame) -> list[str]:
    """Text columns in which every non-blank value parses as a number."""
    cols = []
    for c in X.select_dtypes(include=["object", "string"]).columns:
        vals = unquote(X[c]).replace("", pd.NA).dropna()
        if len(vals) and pd.to_numeric(vals, errors="coerce").notna().all():
            cols.append(c)
    return cols


def coerce_numeric(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cols:
        X[c] = pd.to_numeric(unquote(X[c]), errors="coerce")
    return X


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
    X_train, X_test = train.drop(columns=[target]), test.drop(columns=[target])
    numeric = numeric_text_columns(X_train)  # decided on train only, applied to both
    X_train, cats = to_xgb_frame(coerce_numeric(X_train, numeric))
    X_test, _ = to_xgb_frame(coerce_numeric(X_test, numeric), cats)
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
        "parsed_as_numeric": numeric,
        "models": {"xgboost": {"fit_predict_seconds": round(elapsed, 3), **metrics}},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
