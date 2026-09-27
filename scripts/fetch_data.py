"""Fetch demo datasets into data/ and pre-split them into role-named files.

Every dataset lands as three files (the full dataset itself is never written):
  <name>_history.csv   80%: labeled rows — the ONLY data a model may be fit on
  <name>_test.csv      20%: held-out rows WITH labels — evaluation only
  <name>_current.csv   the SAME 20% rows with the target column removed — what you score

_test and _current are copies of each other (same rows, same order); current just hides
the labels. Scoring "current" therefore scores rows the model never saw in history, and
every scored row can be checked against _test afterwards.

A dataset whose three files already exist is skipped, so re-running never rewrites the
shipped splits. Delete its files to re-download it.

Usage: uv run python scripts/fetch_data.py [name ...]   # default: telco

Names: telco, german_credit, ames_housing, wine_quality, heart_disease
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / "data"

# name -> (source, source_id, filename stem, target column, temporal sort column or None)
DATASETS = {
    "telco": ("openml", 42178, "telco_churn", "Churn", None),
    "german_credit": ("openml", 31, "german_credit", "class", None),
    "ames_housing": ("openml", 42165, "ames_housing", "SalePrice", None),
    "wine_quality": ("uci", 186, "wine_quality", "quality", None),
    "heart_disease": ("uci", 45, "heart_disease", "num", None),
}


def load(name: str) -> pd.DataFrame:
    source, source_id = DATASETS[name][:2]
    if source == "openml":
        from sklearn.datasets import fetch_openml

        df = fetch_openml(data_id=source_id, as_frame=True).frame
    else:  # uci
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            raise SystemExit(f"{name} comes from UCI: run `uv sync --extra datasets` first")

        ds = fetch_ucirepo(id=source_id)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    return df


def split(df: pd.DataFrame, target: str, temporal: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 history/holdout. Temporal data splits by time order, never randomly."""
    if temporal:
        df = df.sort_values(temporal, kind="stable").reset_index(drop=True)
        h = int(len(df) * 0.80)
        return df.iloc[:h], df.iloc[h:]
    stratify = df[target] if df[target].nunique() <= 20 else None
    return train_test_split(df, test_size=0.20, stratify=stratify, random_state=0)


def fetch(name: str) -> None:
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset: {name} (see module docstring)")
    _, _, stem, target, temporal = DATASETS[name]
    paths = [DATA / f"{stem}_{role}.csv" for role in ("history", "test", "current")]
    if all(p.exists() for p in paths):
        print(f"{name}: already in data/ ({', '.join(p.name for p in paths)}), skipped")
        return
    DATA.mkdir(exist_ok=True)
    df = load(name)
    history, holdout = split(df, target, temporal)
    history.to_csv(DATA / f"{stem}_history.csv", index=False)
    holdout.to_csv(DATA / f"{stem}_test.csv", index=False)
    holdout.drop(columns=[target]).to_csv(DATA / f"{stem}_current.csv", index=False)
    print(
        f"{name}: {len(df)} rows -> {stem}_history.csv ({len(history)}), "
        f"{stem}_test.csv ({len(holdout)}), {stem}_current.csv (same {len(holdout)} rows, no '{target}')"
    )


if __name__ == "__main__":
    for dataset in sys.argv[1:] or ["telco"]:
        fetch(dataset)
