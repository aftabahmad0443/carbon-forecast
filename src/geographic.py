import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from baselines import X_COLS, add_log_features, load
from conformal import conf_quantile, winkler
from utils import SEED, set_seed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EPS = 1e-6
K = 5


def make_rf():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_depth=None,
                                 n_jobs=-1, random_state=SEED)


def level(df, yhat):
    return (df["co2"].values + EPS) * np.exp(yhat) - EPS


def interval(df, yhat, q):
    b = df["co2"].values + EPS
    return b * np.exp(yhat - q) - EPS, b * np.exp(yhat + q) - EPS


def log_mae(y, p):
    return np.abs(np.log((np.maximum(p, 0) + EPS) / (y + EPS)))


def main():
    set_seed()
    train, calib, test = (add_log_features(load(n)) for n in ("train", "calib", "test"))

    countries = sorted(train["country"].unique())
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(countries))
    fold_of = {countries[i]: j % K for j, i in enumerate(perm)}
    for d in (train, calib, test):
        d["fold"] = d["country"].map(fold_of)

    # Seen-country reference model: trained on all countries
    rf_all = make_rf().fit(train[X_COLS], train["y"])
    calib["yhat"] = rf_all.predict(calib[X_COLS])
    q_seen = {h: conf_quantile(np.abs(calib.loc[calib["horizon"] == h, "y"]
                                      - calib.loc[calib["horizon"] == h, "yhat"]).values)
              for h in sorted(calib["horizon"].unique())}

    parts = []
    for k in range(K):
        tr = train[train["fold"] != k]
        ca = calib[calib["fold"] != k]
        te = test[test["fold"] == k].copy()
        rf = make_rf().fit(tr[X_COLS], tr["y"])
        s = np.abs(ca["y"] - rf.predict(ca[X_COLS]))
        q_un = {h: conf_quantile(s[ca["horizon"] == h].values)
                for h in sorted(ca["horizon"].unique())}
        te["yhat_unseen"] = rf.predict(te[X_COLS])
        te["yhat_seen"] = rf_all.predict(te[X_COLS])
        te["q_unseen"] = te["horizon"].map(q_un)
        te["q_seen"] = te["horizon"].map(q_seen)
        parts.append(te)
        print(f"fold {k}: {te['country'].nunique()} held-out countries done")
    t = pd.concat(parts, ignore_index=True)

    y = t["target"].values
    p_naive = t["co2"].values
    p_seen = level(t, t["yhat_seen"].values)
    p_un = level(t, t["yhat_unseen"].values)
    lo_s, hi_s = interval(t, t["yhat_seen"].values, t["q_seen"].values)
    lo_u, hi_u = interval(t, t["yhat_unseen"].values, t["q_unseen"].values)
    t["ae_naive"] = log_mae(y, p_naive)
    t["ae_seen"] = log_mae(y, p_seen)
    t["ae_unseen"] = log_mae(y, p_un)
    t["cov_seen"] = (y >= lo_s) & (y <= hi_s)
    t["cov_unseen"] = (y >= lo_u) & (y <= hi_u)
    t["w_seen"] = (hi_s - lo_s) / np.maximum(p_seen, EPS) * 100
    t["w_unseen"] = (hi_u - lo_u) / np.maximum(p_un, EPS) * 100
    t["wink_seen"] = winkler(y, lo_s, hi_s)
    t["wink_unseen"] = winkler(y, lo_u, hi_u)

    g = t.groupby("horizon")
    out = pd.DataFrame({
        "logMAE_naive": g["ae_naive"].mean(),
        "logMAE_seen": g["ae_seen"].mean(),
        "logMAE_unseen": g["ae_unseen"].mean(),
        "cov_seen_%": g["cov_seen"].mean() * 100,
        "cov_unseen_%": g["cov_unseen"].mean() * 100,
        "medrelwidth_seen_%": g["w_seen"].median(),
        "medrelwidth_unseen_%": g["w_unseen"].median(),
        "winkler_seen": g["wink_seen"].mean(),
        "winkler_unseen": g["wink_unseen"].mean(),
    }).round(3)
    print("\nSeen-country vs unseen-country (same held-out countries, test period):")
    print(out.T.to_string())
    out.to_csv(RES / "geographic_metrics.csv")

    print("\nlogMAE by fold, horizon 3 (naive / seen / unseen):")
    print(t[t["horizon"] == 3].groupby("fold")[["ae_naive", "ae_seen", "ae_unseen"]]
          .mean().round(3).to_string())

    print("\nUnseen coverage % by target year (all horizons):")
    print((t.groupby("target_year")["cov_unseen"].mean() * 100).round(1).to_string())

    pk = t[t["country"] == "Pakistan"]
    print("\nPakistan (held out in its fold):")
    print(pk.groupby("horizon")[["ae_naive", "ae_seen", "ae_unseen", "cov_seen",
                                 "cov_unseen"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()