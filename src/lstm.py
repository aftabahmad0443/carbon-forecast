import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from torch import nn

from baselines import X_COLS, add_log_features, load
from conformal import conf_quantile, winkler
from data_prep import (BASE_COLS, START_YEAR, find_end_year,
                       keep_complete_countries, load_countries)
from geographic import interval, level, log_mae, make_rf
from utils import SEED, set_seed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EPS = 1e-6
SEQ = 5
H = np.array([1, 3, 5])
TRAIN_END, VAL_END, CALIB_END = 2008, 2010, 2015
YS = 5.0  # internal target scaling
SEEDS = [SEED, SEED + 1, SEED + 2]


def build_samples():
    df = load_countries()
    end = find_end_year(df, BASE_COLS)
    df = keep_complete_countries(df, BASE_COLS, START_YEAR, end)
    df = df.sort_values(["country", "year"])
    meta, X, Y = [], [], []
    for c, g in df.groupby("country"):
        yrs = g["year"].values
        co2 = g["co2"].values
        pop = g["population"].values
        lco2 = np.log(co2 + EPS)
        feats = np.stack([
            lco2,
            np.log(co2 / pop + EPS),
            np.log(g["gdp"].values / pop),
            np.log(pop),
            np.log(g["energy_per_capita"].values + 1),
        ], axis=1)
        for i in range(SEQ - 1, len(yrs)):
            seq = feats[i - SEQ + 1: i + 1].copy()
            seq[:, 0] -= lco2[i]  # CO2 relative to the current level
            y = [lco2[i + h] - lco2[i] if i + h < len(yrs) else np.nan for h in H]
            meta.append((c, int(yrs[i]), co2[i]))
            X.append(seq)
            Y.append(y)
    meta = pd.DataFrame(meta, columns=["country", "year", "co2"])
    return meta, np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), end


class Net(nn.Module):
    def __init__(self, d=5, hid=32):
        super().__init__()
        self.lstm = nn.LSTM(d, hid, batch_first=True)
        self.head = nn.Linear(hid, len(H))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1])


def masked_l1(pred, y, m):
    return ((pred - y).abs() * m).sum() / m.sum().clamp(min=1)


def train_one(seed, Xa, Ya, Mt, Mv, epochs=150, patience=20):
    torch.manual_seed(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    best = (np.inf, None, 0)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xa))
        for k in range(0, len(perm), 128):
            b = perm[k:k + 128]
            loss = masked_l1(net(Xa[b]), Ya[b], Mt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = masked_l1(net(Xa), Ya, Mv).item()
        if vl < best[0]:
            best = (vl, copy.deepcopy(net.state_dict()), ep)
        elif ep - best[2] >= patience:
            break
    net.load_state_dict(best[1])
    net.eval()
    print(f"  seed {seed}: best epoch {best[2]}, val L1 {best[0] / YS:.4f}")
    return net


def main():
    set_seed()
    torch.set_num_threads(4)
    meta, X, Y, end = build_samples()
    print(f"Samples: {len(meta)}, years up to {end}")

    ty = meta["year"].values[:, None] + H[None, :]
    valid = ~np.isnan(Y)
    m_train = (ty <= TRAIN_END) & valid
    m_val = (ty > TRAIN_END) & (ty <= VAL_END) & valid

    # Standardize inputs using training samples only
    tr_rows = m_train.any(axis=1)
    flat = X[tr_rows].reshape(-1, X.shape[2])
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    Xs = (X - mu) / sd

    Xa = torch.tensor(Xs)
    Ya = torch.tensor(np.nan_to_num(Y) * YS)
    Mt = torch.tensor(m_train.astype(np.float32))
    Mv = torch.tensor(m_val.astype(np.float32))

    print("Training LSTM ensemble:")
    nets = [train_one(s, Xa, Ya, Mt, Mv) for s in SEEDS]
    with torch.no_grad():
        yhat_all = np.mean([n(Xa).numpy() for n in nets], axis=0) / YS

    # Long format for calibration and test rows
    parts = []
    for j, h in enumerate(H):
        d = meta.copy()
        d["horizon"] = h
        d["target_year"] = d["year"] + h
        d["y"] = Y[:, j]
        d["yhat_lstm"] = yhat_all[:, j]
        parts.append(d[valid[:, j]])
    lg = pd.concat(parts, ignore_index=True)
    calib_l = lg[(lg["target_year"] > VAL_END + 1 - 1) & (lg["target_year"] > 2010)
                 & (lg["target_year"] <= CALIB_END)]
    test_l = lg[lg["target_year"] > CALIB_END]

    # RF reference on the same pipeline
    train, calib, test = (add_log_features(load(n)) for n in ("train", "calib", "test"))
    rf = make_rf().fit(train[X_COLS], train["y"])
    for d in (calib, test):
        d["yhat_rf"] = rf.predict(d[X_COLS])

    keys = ["country", "year", "horizon"]
    t = test[keys + ["target_year", "target", "co2", "y", "yhat_rf"]].merge(
        test_l[keys + ["yhat_lstm", "y"]], on=keys, suffixes=("", "_l"))
    print(f"Test rows (common): {len(t)} of RF {len(test)}")
    print("Max |y_rf - y_lstm| (target consistency check):",
          float(np.abs(t["y"] - t["y_l"]).max()))

    # Per-horizon conformal quantiles from each model's own calibration scores
    q_rf, q_ls = {}, {}
    for h in H:
        c_rf = calib[calib["horizon"] == h]
        q_rf[h] = conf_quantile(np.abs(c_rf["y"] - c_rf["yhat_rf"]).values)
        c_l = calib_l[calib_l["horizon"] == h]
        q_ls[h] = conf_quantile(np.abs(c_l["y"] - c_l["yhat_lstm"]).values)
    print("Quantiles RF  :", {int(h): round(v, 4) for h, v in q_rf.items()})
    print("Quantiles LSTM:", {int(h): round(v, 4) for h, v in q_ls.items()})
    t["q_rf"] = t["horizon"].map(q_rf)
    t["q_lstm"] = t["horizon"].map(q_ls)

    y = t["target"].values
    t["ae_naive"] = log_mae(y, t["co2"].values)
    for m in ("rf", "lstm"):
        yh = t[f"yhat_{m}"].values
        p = level(t, yh)
        lo, hi = interval(t, yh, t[f"q_{m}"].values)
        t[f"ae_{m}"] = log_mae(y, p)
        t[f"cov_{m}"] = (y >= lo) & (y <= hi)
        t[f"w_{m}"] = (hi - lo) / np.maximum(p, EPS) * 100
        t[f"wink_{m}"] = winkler(y, lo, hi)

    g = t.groupby("horizon")
    out = pd.DataFrame({
        "logMAE_naive": g["ae_naive"].mean(),
        "logMAE_rf": g["ae_rf"].mean(),
        "logMAE_lstm": g["ae_lstm"].mean(),
        "cov_rf_%": g["cov_rf"].mean() * 100,
        "cov_lstm_%": g["cov_lstm"].mean() * 100,
        "medrelwidth_rf_%": g["w_rf"].median(),
        "medrelwidth_lstm_%": g["w_lstm"].median(),
        "winkler_rf": g["wink_rf"].mean(),
        "winkler_lstm": g["wink_lstm"].mean(),
    }).round(3)
    print("\nTest period comparison (target coverage 90%):")
    print(out.T.to_string())
    out.to_csv(RES / "lstm_metrics.csv")

    print("\nCoverage % by target year (all horizons):")
    cy = pd.DataFrame({m: t.groupby("target_year")[f"cov_{m}"].mean() * 100
                       for m in ("rf", "lstm")}).round(1)
    print(cy.to_string())

    print("\nlogMAE by target year (h=1):")
    print(t[t["horizon"] == 1].groupby("target_year")[["ae_naive", "ae_rf", "ae_lstm"]]
          .mean().round(3).to_string())

    pk = t[t["country"] == "Pakistan"]
    print("\nPakistan:")
    print(pk.groupby("horizon")[["ae_naive", "ae_rf", "ae_lstm", "cov_rf", "cov_lstm"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()