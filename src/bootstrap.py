from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
B = 5000
SEED = 42

COLS = ["ae_naive", "ae_rf", "ae_lstm", "cov_rf", "cov_lstm"]


def main():
    t = pd.read_csv(RES / "lstm_test_rows.csv")
    t[["cov_rf", "cov_lstm"]] = t[["cov_rf", "cov_lstm"]].astype(float)
    rng = np.random.default_rng(SEED)
    rows = []

    for h in sorted(t["horizon"].unique()):
        d = t[t["horizon"] == h]
        agg = d.groupby("country")[COLS].sum()
        n = d.groupby("country").size().values.astype(float)
        S = agg.values                      # countries x metrics (sums)
        k = len(agg)
        idx = rng.integers(0, k, size=(B, k))
        tot = S[idx].sum(axis=1)            # B x metrics
        cnt = n[idx].sum(axis=1)[:, None]
        m = tot / cnt                       # bootstrap means
        point = S.sum(axis=0) / n.sum()

        def ci(v):
            return np.percentile(v, [2.5, 97.5])

        c = {name: i for i, name in enumerate(COLS)}
        stats = {
            "RF - Naive (logMAE)": (point[c["ae_rf"]] - point[c["ae_naive"]],
                                    m[:, c["ae_rf"]] - m[:, c["ae_naive"]]),
            "LSTM - Naive (logMAE)": (point[c["ae_lstm"]] - point[c["ae_naive"]],
                                      m[:, c["ae_lstm"]] - m[:, c["ae_naive"]]),
            "LSTM - RF (logMAE)": (point[c["ae_lstm"]] - point[c["ae_rf"]],
                                   m[:, c["ae_lstm"]] - m[:, c["ae_rf"]]),
            "RF coverage %": (point[c["cov_rf"]] * 100, m[:, c["cov_rf"]] * 100),
            "LSTM coverage %": (point[c["cov_lstm"]] * 100, m[:, c["cov_lstm"]] * 100),
        }
        for name, (pt, boot) in stats.items():
            lo, hi = ci(boot)
            rows.append({"horizon": h, "quantity": name, "estimate": pt,
                         "ci_low": lo, "ci_high": hi,
                         "excludes_0/90": (lo > 0 or hi < 0) if "logMAE" in name
                         else (lo > 90 or hi < 90)})

    out = pd.DataFrame(rows).round(4)
    print(f"Country-cluster bootstrap, B={B}, 95% percentile intervals")
    print("(logMAE differences: negative = first model better; "
          "coverage flag: True = interval excludes 90)\n")
    print(out.to_string(index=False))
    out.to_csv(RES / "bootstrap_ci.csv", index=False)


if __name__ == "__main__":
    main()