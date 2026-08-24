"""Fetch the demo dataset bundle into data/ (no credentials needed).

Usage: python3 scripts/fetch_data.py [name ...]   # default: telco

Names: telco, german_credit, ames_housing, wine_quality, heart_disease, premier_league
"""

import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

OPENML = {  # name -> (openml data_id, filename)
    "telco": (42178, "telco_churn.csv"),
    "german_credit": (31, "german_credit.csv"),
    "ames_housing": (42165, "ames_housing.csv"),
}
UCI = {  # name -> (ucimlrepo id, filename)
    "wine_quality": (186, "wine_quality.csv"),
    "heart_disease": (45, "heart_disease.csv"),
}
# football-data.co.uk season codes for the last 10 Premier League seasons
PL_SEASONS = ["1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]


def fetch(name: str) -> Path:
    DATA.mkdir(exist_ok=True)
    if name in OPENML:
        from sklearn.datasets import fetch_openml

        data_id, filename = OPENML[name]
        df = fetch_openml(data_id=data_id, as_frame=True).frame
    elif name in UCI:
        from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo

        uci_id, filename = UCI[name]
        ds = fetch_ucirepo(id=uci_id)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    elif name == "premier_league":
        filename = "premier_league_2015_2025.csv"
        frames = []
        for season in PL_SEASONS:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
            frame = pd.read_csv(url)
            frame["Season"] = season
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
    else:
        raise SystemExit(f"unknown dataset: {name} (see module docstring)")
    out = DATA / filename
    df.to_csv(out, index=False)
    print(f"{name}: {df.shape[0]} rows x {df.shape[1]} cols -> {out}")
    return out


if __name__ == "__main__":
    for dataset in sys.argv[1:] or ["telco"]:
        fetch(dataset)
