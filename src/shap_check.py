import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from baselines import X_COLS, add_log_features, load
from explain import GROUPS
from utils import SEED, set_seed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def main():
    set_seed()
    train, _, test = (add_log_features(load(n)) for n in ("train", "calib", "test"))
    test = test.reset_index(drop=True)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_depth=None,
                               n_jobs=-1, random_state=SEED).fit(train[X_COLS], train["y"])

    ex = shap.TreeExplainer(rf)
    X = test[X_COLS]
    sv = ex.shap_values(X, check_additivity=False)
    base = float(np.ravel(ex.expected_value)[0])

    # 1. Additivity: base + sum(SHAP) should equal the model prediction
    diff = np.abs(base + sv.sum(axis=1) - rf.predict(X))
    print(f"Additivity check: max abs diff = {diff.max():.2e}, mean = {diff.mean():.2e}")

    # 2. Importance without the mechanical 'horizon' feature
    abs_df = pd.DataFrame(np.abs(sv), columns=X_COLS).drop(columns="horizon")
    tab = pd.DataFrame({"all": abs_df.mean()})
    for h in sorted(test["horizon"].unique()):
        tab[f"h={h}"] = abs_df[test["horizon"] == h].mean()
    share = (tab / tab.sum() * 100).round(1).sort_values("all", ascending=False)
    print("\nFeature share %, horizon excluded:")
    print(share.to_string())

    groups = {g: [c for c in cols if c != "horizon"] for g, cols in GROUPS.items()}
    groups.pop("Horizon")
    g = pd.DataFrame({k: abs_df[v].sum(axis=1) for k, v in groups.items()})
    gt = pd.DataFrame({"all": g.mean()})
    for h in sorted(test["horizon"].unique()):
        gt[f"h={h}"] = g[test["horizon"] == h].mean()
    gshare = (gt / gt.sum() * 100).round(1)
    print("\nGrouped share %, horizon excluded:")
    print(gshare.to_string())
    share.to_csv(RES / "shap_share_no_horizon.csv")
    gshare.to_csv(RES / "shap_grouped_no_horizon.csv")


if __name__ == "__main__":
    main()