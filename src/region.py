import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from pycountry_convert import (country_alpha2_to_continent_code,
                               country_alpha3_to_country_alpha2)

from baselines import X_COLS, add_log_features, load
from conformal import conf_quantile
from geographic import interval, level, log_mae, make_rf
from utils import set_seed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EPS = 1e-6
NAMES = {"AF": "Africa", "AS": "Asia", "EU": "Europe",
         "NA": "North America", "SA": "South America", "OC": "Oceania"}


def continent(iso3):
    try:
        return NAMES[country_alpha2_to_continent_code(
            country_alpha3_to_country_alpha2(iso3))]
    except Exception:
        return None


def q_by_horizon(ca, yhat):
    s = np.abs(ca["y"].values - yhat)
    return {h: conf_quantile(s[(ca["horizon"] == h).values])
            for h in sorted(ca["horizon"].unique())}


def main():
    set_seed()
    train, calib, test = (add_log_features(load(n)) for n in ("train", "calib", "test"))
    for d in (train, calib, test):
        d["region"] = d["iso_code"].map(continent)

    unmapped = sorted(set(train.loc[train["region"].isna(), "country"]))
    print("Unmapped countries (dropped):", unmapped)
    train, calib, test = (d[d["region"].notna()].copy() for d in (train, calib, test))
    print(test.groupby("region")["country"].nunique().rename("n_countries").to_string())

    rf_all = make_rf().fit(train[X_COLS], train["y"])
    q_seen = q_by_horizon(calib, rf_all.predict(calib[X_COLS]))

    parts = []
    for r in sorted(test["region"].unique()):
        tr, ca = train[train["region"] != r], calib[calib["region"] != r]
        te = test[test["region"] == r].copy()
        rf = make_rf().fit(tr[X_COLS], tr["y"])
        q_un = q_by_horizon(ca, rf.predict(ca[X_COLS]))
        te["yhat_un"] = rf.predict(te[X_COLS])
        te["yhat_seen"] = rf_all.predict(te[X_COLS])
        te["q_un"] = te["horizon"].map(q_un)
        te["q_seen"] = te["horizon"].map(q_seen)
        parts.append(te)
        print(f"held out {r}: done")
    t = pd.concat(parts, ignore_index=True)

    y = t["target"].values
    lo_s, hi_s = interval(t, t["yhat_seen"].values, t["q_seen"].values)
    lo_u, hi_u = interval(t, t["yhat_un"].values, t["q_un"].values)
    t["ae_naive"] = log_mae(y, t["co2"].values)
    t["ae_seen"] = log_mae(y, level(t, t["yhat_seen"].values))
    t["ae_unseen"] = log_mae(y, level(t, t["yhat_un"].values))
    t["cov_seen"] = (y >= lo_s) & (y <= hi_s)
    t["cov_unseen"] = (y >= lo_u) & (y <= hi_u)

    cols = ["ae_naive", "ae_seen", "ae_unseen", "cov_seen", "cov_unseen"]
    out = t.groupby(["region", "horizon"])[cols].mean()
    out[["cov_seen", "cov_unseen"]] *= 100
    out = out.round(3)
    print("\nLeave-one-continent-out (logMAE and coverage %):")
    print(out.to_string())
    out.to_csv(RES / "region_metrics.csv")

    allh = t.groupby("horizon")[cols].mean()
    allh[["cov_seen", "cov_unseen"]] *= 100
    print("\nAll continents pooled:")
    print(allh.round(3).to_string())

    pk = t[t["country"] == "Pakistan"]
    print("\nPakistan (Asia held out):")
    print(pk.groupby("horizon")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()