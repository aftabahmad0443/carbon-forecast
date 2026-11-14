import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from baselines import X_COLS, add_log_features, load
from utils import SEED, set_seed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EPS = 1e-6
ALPHA = 0.10  # target coverage 90%
MIN_POOL = 100  # min calibration scores for rolling version


def conf_quantile(scores, alpha=ALPHA):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def winkler(y, lo, hi, alpha=ALPHA):
    w = hi - lo
    w = w + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)
    return w


def summarize(df, lo, hi, label):
    y = df["target"].values
    cover = (y >= lo) & (y <= hi)
    out = {
        "method": label,
        "coverage_%": cover.mean() * 100,
        "mean_width_Mt": np.mean(hi - lo),
        "median_rel_width_%": np.median((hi - lo) / np.maximum(df["pred"].values, EPS)) * 100,
        "winkler": np.mean(winkler(y, lo, hi)),
    }
    return out, cover


def main():
    set_seed()
    train, calib, test = (add_log_features(load(n)) for n in ("train", "calib", "test"))

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_depth=None,
                               n_jobs=-1, random_state=SEED).fit(train[X_COLS], train["y"])

    for d in (calib, test):
        d["yhat"] = rf.predict(d[X_COLS])
        d["score"] = np.abs(d["y"] - d["yhat"])
        d["pred"] = (d["co2"] + EPS) * np.exp(d["yhat"]) - EPS

    # Static split conformal: one quantile per horizon from the calibration set
    q_static = {h: conf_quantile(calib.loc[calib["horizon"] == h, "score"].values)
                for h in sorted(calib["horizon"].unique())}
    print("Static log-scale quantiles:", {h: round(q, 4) for h, q in q_static.items()})

    # Rolling split conformal: only scores whose target is already observed
    pool = pd.concat([calib, test], ignore_index=True)
    q_roll = np.zeros(len(test))
    for (h, y0), idx in test.groupby(["horizon", "year"]).groups.items():
        known = pool[(pool["horizon"] == h) & (pool["target_year"] <= y0)]["score"].values
        if len(known) < MIN_POOL:
            known = calib.loc[calib["horizon"] == h, "score"].values
        q_roll[test.index.get_indexer(idx)] = conf_quantile(known)

    base = (test["co2"].values + EPS)
    results = {
        "Static": q_static_arr if (q_static_arr := test["horizon"].map(q_static).values) is not None else None,
        "Rolling": q_roll,
    }

    rows, cover_by = [], {}
    for name, q in results.items():
        lo = base * np.exp(test["yhat"].values - q) - EPS
        hi = base * np.exp(test["yhat"].values + q) - EPS
        for h in sorted(test["horizon"].unique()):
            m = (test["horizon"] == h).values
            r, cov = summarize(test[m], lo[m], hi[m], name)
            rows.append({"horizon": h, **r})
        r, cov = summarize(test, lo, hi, name)
        cover_by[name] = pd.Series(cov, index=test.index)

    out = pd.DataFrame(rows).round(2)
    print("\nTarget coverage: 90%")
    print(out.sort_values(["horizon", "method"]).to_string(index=False))
    out.to_csv(RES / "conformal_metrics.csv", index=False)

    print("\nCoverage % by target year (all horizons):")
    cy = pd.DataFrame({n: s.groupby(test["target_year"]).mean() * 100
                       for n, s in cover_by.items()}).round(1)
    print(cy.to_string())
    cy.to_csv(RES / "conformal_coverage_by_year.csv")

    print("\nCoverage % by horizon x target year, Static:")
    ct = (cover_by["Static"].groupby([test["horizon"], test["target_year"]]).mean()
          .unstack() * 100).round(0)
    print(ct.to_string())

    # Save intervals for the Streamlit demo
    keep = test[["country", "year", "horizon", "target_year", "target", "pred"]].copy()
    keep["lo_static"] = base * np.exp(test["yhat"].values - test["horizon"].map(q_static).values) - EPS
    keep["hi_static"] = base * np.exp(test["yhat"].values + test["horizon"].map(q_static).values) - EPS
    keep["lo_roll"] = base * np.exp(test["yhat"].values - q_roll) - EPS
    keep["hi_roll"] = base * np.exp(test["yhat"].values + q_roll) - EPS
    keep.to_csv(RES / "conformal_test_intervals.csv", index=False)


if __name__ == "__main__":
    main()