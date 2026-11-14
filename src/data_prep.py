from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/owid-co2-data.csv")
OUT = Path("data/processed")
START_YEAR = 1990
HORIZONS = [1, 3, 5]
BASE_COLS = ["co2", "population", "gdp", "energy_per_capita"]

# Chronological split by TARGET year (prevents leakage across periods)
TRAIN_END = 2010
CALIB_END = 2015  # calibration: 2011-2015, test: 2016 onwards


def load_countries() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    # Real countries only: have an ISO code, and are not OWID aggregates
    df = df[df["iso_code"].notna() & ~df["iso_code"].str.startswith("OWID")]
    return df.copy()


def find_end_year(df: pd.DataFrame, cols, min_countries: int = 100) -> int:
    """Last year in which every required column has data for many countries."""
    ends = []
    for c in cols:
        n = df.dropna(subset=[c]).groupby("year")["country"].nunique()
        ends.append(int(n[n >= min_countries].index.max()))
        print(f"  last year with >= {min_countries} countries for {c}: {ends[-1]}")
    return min(ends)


def keep_complete_countries(df, cols, start, end) -> pd.DataFrame:
    d = df[(df["year"] >= start) & (df["year"] <= end)]
    n_years = end - start + 1
    ok = (
        d.dropna(subset=cols)
        .groupby("country")["year"]
        .nunique()
    )
    complete = ok[ok == n_years].index
    dropped = sorted(set(d["country"]) - set(complete))
    print(f"Countries kept: {len(complete)}, dropped: {len(dropped)}")
    return d[d["country"].isin(complete)].copy()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    g = df.groupby("country")["co2"]

    for k in (1, 2, 3):
        df[f"co2_lag{k}"] = g.shift(k)
    df["co2_growth"] = (df["co2"] / df["co2_lag1"] - 1).replace([np.inf, -np.inf], np.nan)
    df["co2_roll3"] = g.transform(lambda s: s.rolling(3).mean())
    df["gdp_per_capita"] = df["gdp"] / df["population"]
    df["log_gdp_pc"] = np.log(df["gdp_per_capita"])
    df["log_population"] = np.log(df["population"])
    df["co2_per_capita"] = df["co2"] / df["population"]
    return df


FEATURES = [
    "co2", "co2_lag1", "co2_lag2", "co2_lag3", "co2_growth", "co2_roll3",
    "log_gdp_pc", "log_population", "energy_per_capita", "co2_per_capita",
]


def make_long(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for h in HORIZONS:
        d = df.copy()
        d["horizon"] = h
        d["target"] = d.groupby("country")["co2"].shift(-h)
        d["target_year"] = d["year"] + h
        parts.append(d)
    long = pd.concat(parts, ignore_index=True)
    long = long.dropna(subset=FEATURES + ["target"])
    return long


def split(long: pd.DataFrame):
    ty = long["target_year"]
    train = long[ty <= TRAIN_END]
    calib = long[(ty > TRAIN_END) & (ty <= CALIB_END)]
    test = long[ty > CALIB_END]
    return train, calib, test


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_countries()
    print("Checking data coverage:")
    end_year = find_end_year(df, BASE_COLS)
    print(f"Using years {START_YEAR}-{end_year}")

    df = keep_complete_countries(df, BASE_COLS, START_YEAR, end_year)
    print("Pakistan included:", "Pakistan" in set(df["country"]))

    df = make_features(df)
    long = make_long(df)
    train, calib, test = split(long)

    keep = ["country", "iso_code", "year", "horizon", "target_year", "target"] + FEATURES
    for name, part in [("train", train), ("calib", calib), ("test", test), ("all", long)]:
        part[keep].to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows, target years "
              f"{part['target_year'].min()}-{part['target_year'].max()}")


if __name__ == "__main__":
    main()