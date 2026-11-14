import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from baselines import X_COLS, add_log_features, load
from conformal import winkler
from geographic import interval, make_rf
from region import continent
from utils import set_seed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EPS = 1e-6
ALPHA = 0.10
MIN_N = 50       # min calibration scores per (region, horizon) for a group quantile
W_CLIP = 20.0    # clip density-ratio weights to [1/W_CLIP, W_CLIP]
FEATS = [c for c in X_COLS if c != "horizon"]


def cq(scores, alpha=ALPHA):
    """Exact split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    s = np.sort(np.asarray(scores, dtype=float))
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[min(k, len(s)) - 1])


def wquantile(scores, w, w_test, alpha=ALPHA):
    """Weighted conformal quantile (Tibshirani et al. 2019) for each test weight.
    Returns (quantile, was_infinite); infinite quantiles are capped at max(score)."""
    o = np.argsort(scores)
    s, ww = scores[o], w[o]
    cum = np.cumsum(ww)
    target = (1 - alpha) * (ww.sum() + np.atleast_1d(w_test))
    idx = np.searchsorted(cum, target - 1e-12, side="left")
    inf = idx >= len(s)
    return s[np.minimum(idx, len(s) - 1)], inf


def evaluate(df, q):
    y, yh = df["target"].values, df["yhat"].values
    lo, hi = interval(df, yh, q)
    pred = (df["co2"].values + EPS) * np.exp(yh) - EPS
    cov = (y >= lo) & (y <= hi)
    return cov, (hi - lo) / np.maximum(pred, EPS) * 100, winkler(y, lo, hi)


def prep():
    train, calib, test = (add_log_features(load(n)) for n in ("train", "calib", "test"))
    for d in (train, calib, test):
        d["region"] = d["iso_code"].map(continent)
    return tuple(d[d["region"].notna()].reset_index(drop=True) for d in (train, calib, test))


def part_a(train, calib, test):
    """Seen continents: one pooled quantile per horizon vs one per (continent, horizon)."""
    print("=" * 70)
    print("PART A: Mondrian (continent-wise) vs pooled calibration, temporal test")
    rf = make_rf().fit(train[X_COLS], train["y"])
    for d in (calib, test):
        d["yhat"] = rf.predict(d[X_COLS])
    calib["score"] = np.abs(calib["y"] - calib["yhat"])
    hs = sorted(calib["horizon"].unique())

    q_pool = {h: cq(calib.loc[calib["horizon"] == h, "score"].values) for h in hs}
    q_grp, pooled_fallback = {}, set()
    for (r, h), g in calib.groupby(["region", "horizon"]):
        if len(g) >= MIN_N:
            q_grp[(r, h)] = cq(g["score"].values)
        else:
            q_grp[(r, h)] = q_pool[h]
            pooled_fallback.add(r)
    print("Regions falling back to the pooled quantile (too few calibration scores):",
          sorted(pooled_fallback))
    print("Calibration scores per region, horizon 3:",
          calib[calib["horizon"] == 3].groupby("region").size().to_dict())

    q_std = test["horizon"].map(q_pool).values
    q_mon = np.array([q_grp[(r, h)] for r, h in zip(test["region"], test["horizon"])])
    out = {}
    for name, q in (("Pooled", q_std), ("Mondrian", q_mon)):
        out[name] = evaluate(test, q)

    rows = []
    for h in hs:
        m = (test["horizon"] == h).values
        for name, (cov, w, wk) in out.items():
            rows.append({"horizon": h, "method": name, "coverage_%": cov[m].mean() * 100,
                         "med_rel_width_%": np.median(w[m]), "winkler": wk[m].mean()})
    summ = pd.DataFrame(rows).round(2)
    print("\nPooled over all test countries (target 90%):")
    print(summ.to_string(index=False))
    summ.to_csv(RES / "mondrian_metrics.csv", index=False)

    reg = pd.DataFrame({"region": test["region"], "horizon": test["horizon"],
                        "Pooled": out["Pooled"][0], "Mondrian": out["Mondrian"][0]})
    reg_t = (reg.groupby(["region", "horizon"])[["Pooled", "Mondrian"]].mean() * 100).round(1)
    print("\nCoverage % by continent and horizon:")
    print(reg_t.to_string())
    reg_t.to_csv(RES / "mondrian_by_region.csv")

    print("\nWorst continent gap |coverage - 90| (continents with >= 20 countries):")
    big = test.groupby("region")["country"].nunique()
    big = big[big >= 20].index
    for name in ("Pooled", "Mondrian"):
        gaps = (reg[reg["region"].isin(big)].groupby(["region", "horizon"])[name].mean() * 100 - 90).abs()
        print(f"  {name}: max {gaps.max():.1f} points, mean {gaps.mean():.1f} points")


def part_b(train, calib, test):
    """Unseen continent: pooled calibration vs weighted conformal (covariate shift)."""
    print("\n" + "=" * 70)
    print("PART B: leave-one-continent-out, pooled vs weighted conformal")
    hs = sorted(test["horizon"].unique())
    parts = []
    for r in sorted(test["region"].unique()):
        tr = train[train["region"] != r]
        ca = calib[calib["region"] != r].copy()
        te = test[test["region"] == r].copy().reset_index(drop=True)
        rf = make_rf().fit(tr[X_COLS], tr["y"])
        ca["yhat"], te["yhat"] = rf.predict(ca[X_COLS]), rf.predict(te[X_COLS])
        ca["score"] = np.abs(ca["y"] - ca["yhat"])

        sc = StandardScaler().fit(pd.concat([ca[FEATS], te[FEATS]]))
        Xc, Xt = sc.transform(ca[FEATS]), sc.transform(te[FEATS])
        clf = LogisticRegression(max_iter=2000).fit(
            np.vstack([Xc, Xt]), np.r_[np.zeros(len(Xc)), np.ones(len(Xt))])
        scale = len(Xc) / len(Xt)  # undo class-size imbalance in the odds
        pc, pt = clf.predict_proba(Xc), clf.predict_proba(Xt)
        wc = np.clip(pc[:, 1] / pc[:, 0] * scale, 1 / W_CLIP, W_CLIP)
        wt = np.clip(pt[:, 1] / pt[:, 0] * scale, 1 / W_CLIP, W_CLIP)
        ess = wc.sum() ** 2 / (wc ** 2).sum() / len(wc)

        q_u, q_w, capped = np.zeros(len(te)), np.zeros(len(te)), np.zeros(len(te), bool)
        for h in hs:
            mc, mt = (ca["horizon"] == h).values, (te["horizon"] == h).values
            s = ca["score"].values[mc]
            q_u[mt] = cq(s)
            qq, inf = wquantile(s, wc[mc], wt[mt])
            q_w[mt], capped[mt] = qq, inf
        for name, q in (("Pooled", q_u), ("Weighted", q_w)):
            cov, wd, wk = evaluate(te, q)
            te[f"cov_{name}"], te[f"w_{name}"], te[f"wk_{name}"] = cov, wd, wk
        te["capped"] = capped
        parts.append(te)
        print(f"held out {r}: ESS/n = {ess:.2f}, capped intervals = {capped.mean() * 100:.1f}%")

    t = pd.concat(parts, ignore_index=True)
    g = t.groupby(["region", "horizon"])
    tab = pd.DataFrame({
        "cov_Pooled": g["cov_Pooled"].mean() * 100, "cov_Weighted": g["cov_Weighted"].mean() * 100,
        "w_Pooled": g["w_Pooled"].median(), "w_Weighted": g["w_Weighted"].median()}).round(1)
    print("\nUnseen-continent coverage % and median relative width %:")
    print(tab.to_string())
    tab.to_csv(RES / "weighted_loco.csv")

    gh = t.groupby("horizon")
    pooled = pd.DataFrame({
        "cov_Pooled": gh["cov_Pooled"].mean() * 100, "cov_Weighted": gh["cov_Weighted"].mean() * 100,
        "w_Pooled": gh["w_Pooled"].median(), "w_Weighted": gh["w_Weighted"].median(),
        "winkler_Pooled": gh["wk_Pooled"].mean(), "winkler_Weighted": gh["wk_Weighted"].mean()}).round(2)
    print("\nAll continents pooled (target 90%):")
    print(pooled.to_string())


def main():
    set_seed()
    train, calib, test = prep()
    part_a(train, calib, test)
    part_b(train, calib, test)


if __name__ == "__main__":
    main()