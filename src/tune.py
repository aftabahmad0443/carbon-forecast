import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from baselines import X_COLS, add_log_features, load, metrics, to_level
from utils import SEED, set_seed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EPS = 1e-6

# Validation folds by target year; always train on earlier target years only
FOLDS = [(2002, 2004), (2005, 2007), (2008, 2010)]


def cv_score(make_model, train):
    errs = []
    for lo, hi in FOLDS:
        tr = train[train["target_year"] < lo]
        va = train[(train["target_year"] >= lo) & (train["target_year"] <= hi)]
        m = make_model().fit(tr[X_COLS], tr["y"])
        errs.append(np.abs(va["y"] - m.predict(va[X_COLS])).mean())
    return float(np.mean(errs))


def tune(name, make, grid, train):
    keys = list(grid)
    best = (np.inf, None)
    for vals in itertools.product(*grid.values()):
        params = dict(zip(keys, vals))
        s = cv_score(lambda: make(**params), train)
        print(f"  {name} {params} -> CV logMAE {s:.4f}")
        if s < best[0]:
            best = (s, params)
    print(f"Best {name}: {best[1]} (CV logMAE {best[0]:.4f})\n")
    return best[1]


def drift_pred(df):
    ratio = ((df["co2"] + EPS) / (df["co2_lag3"] + EPS)).clip(0.5, 2.0)
    return df["co2"].values * np.power(ratio.values, df["horizon"].values / 3.0)


def log_mae(y, p):
    return float(np.mean(np.abs(np.log((np.maximum(p, 0) + EPS) / (y + EPS)))))


def main():
    set_seed()
    train, calib, test = (add_log_features(load(n)) for n in ("train", "calib", "test"))

    print("Naive CV logMAE:", round(float(np.abs(train["y"]).mean()), 4), "\n")

    rf_grid = {"min_samples_leaf": [5, 20, 50], "max_depth": [4, 8, None]}
    xgb_grid = {"max_depth": [2, 3, 4], "n_estimators": [100, 300],
                "min_child_weight": [5, 30]}

    rf_best = tune("RF", lambda **p: RandomForestRegressor(
        n_estimators=200, n_jobs=-1, random_state=SEED, **p), rf_grid, train)
    xgb_best = tune("XGB", lambda **p: XGBRegressor(
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, **p), xgb_grid, train)

    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED,
                               **rf_best).fit(train[X_COLS], train["y"])
    xgb = XGBRegressor(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                       random_state=SEED, n_jobs=-1, **xgb_best
                       ).fit(train[X_COLS], train["y"])

    preds = {
        "Naive": test["co2"].values,
        "Drift (3y)": drift_pred(test),
        "RF tuned": to_level(test, rf.predict(test[X_COLS])),
        "XGB tuned": to_level(test, xgb.predict(test[X_COLS])),
    }

    rows = []
    for h in sorted(test["horizon"].unique()):
        mask = (test["horizon"] == h).values
        y = test["target"].values[mask]
        for name, p in preds.items():
            p = p[mask]
            m = metrics(y, p)
            rows.append({"horizon": h, "model": name, "MAE": m["MAE"],
                         "logMAE": log_mae(y, p),
                         "medAPE_%": float(np.median(np.abs(y - p) / np.maximum(y, EPS)) * 100)})
    out = pd.DataFrame(rows).round(3)
    print(out.to_string(index=False))
    out.to_csv(RES / "tuned_metrics.csv", index=False)


if __name__ == "__main__":
    main()