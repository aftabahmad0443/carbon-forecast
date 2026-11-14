import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from baselines import X_COLS, add_log_features, load
from utils import SEED, set_seed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"

GROUPS = {
    "Momentum (lags, growth, roll3)": ["lag1_ratio", "lag2_ratio", "lag3_ratio",
                                       "roll3_ratio", "co2_growth_c"],
    "Scale (CO2, CO2 pc, population)": ["log_co2", "log_co2_pc", "log_population"],
    "Economy/energy (GDP pc, energy pc)": ["log_gdp_pc", "log_energy_pc"],
    "Horizon": ["horizon"],
}
COUNTRIES = ["Pakistan", "China", "India", "United States", "Germany", "Brazil"]
CASES = [("Pakistan", 2019, 1), ("China", 2019, 1), ("Pakistan", 2021, 1)]


def share(abs_df):
    """Mean |SHAP| as % of total, per column of abs_df (rows=features)."""
    return abs_df / abs_df.sum() * 100


def grouped(sv_df):
    return pd.DataFrame({g: sv_df[cols].abs().sum(axis=1) for g, cols in GROUPS.items()})


def main():
    set_seed()
    FIG.mkdir(parents=True, exist_ok=True)
    train, _, test = (add_log_features(load(n)) for n in ("train", "calib", "test"))
    test = test.reset_index(drop=True)

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_depth=None,
                               n_jobs=-1, random_state=SEED).fit(train[X_COLS], train["y"])

    print("Computing SHAP values (can take a few minutes)...")
    explainer = shap.TreeExplainer(rf)
    X = test[X_COLS]
    sv = explainer.shap_values(X, check_additivity=False)
    base = float(np.ravel(explainer.expected_value)[0])
    sv_df = pd.DataFrame(sv, columns=X_COLS)
    abs_df = sv_df.abs()
    print(f"Base value (mean predicted log-ratio): {base:.4f}")

    # 1. Global importance: overall and per horizon
    glob = pd.DataFrame({"all": abs_df.mean()})
    for h in sorted(test["horizon"].unique()):
        glob[f"h={h}"] = abs_df[test["horizon"] == h].mean()
    glob = glob.sort_values("all", ascending=False)
    print("\nMean |SHAP| (log-ratio units):")
    print(glob.round(4).to_string())
    glob.round(5).to_csv(RES / "shap_global_importance.csv")

    ax = glob["all"].sort_values().plot.barh(figsize=(7, 5))
    ax.set_xlabel("Mean |SHAP| (log-ratio units)")
    ax.set_title("Global feature importance (tuned RF, test set)")
    plt.tight_layout(); plt.savefig(FIG / "shap_bar.png", dpi=150); plt.close()

    plt.figure()
    shap.summary_plot(sv, X, show=False)
    plt.tight_layout(); plt.savefig(FIG / "shap_beeswarm.png", dpi=150); plt.close()

    # 2. Grouped importance (more stable under correlated features)
    g_all = grouped(sv_df)
    g_tab = pd.DataFrame({"all": g_all.mean()})
    for h in sorted(test["horizon"].unique()):
        g_tab[f"h={h}"] = g_all[test["horizon"] == h].mean()
    g_share = share(g_tab).round(1)
    print("\nGrouped importance, % share:")
    print(g_share.to_string())
    g_share.to_csv(RES / "shap_grouped_share.csv")

    # 3. Over time: 2016-2019 vs 2020-2022 (by target year)
    period = np.where(test["target_year"] <= 2019, "2016-2019", "2020-2022")
    per = pd.DataFrame({p: abs_df[period == p].mean() for p in ("2016-2019", "2020-2022")})
    per_g = pd.DataFrame({p: g_all[period == p].mean() for p in ("2016-2019", "2020-2022")})
    print("\nFeature share % by period:")
    print(share(per).round(1).sort_values("2016-2019", ascending=False).to_string())
    print("\nGrouped share % by period:")
    print(share(per_g).round(1).to_string())
    share(per).round(2).to_csv(RES / "shap_by_period.csv")

    # 4. Across countries (horizon 3 as a middle case)
    rows = {}
    for c in COUNTRIES:
        m = ((test["country"] == c) & (test["horizon"] == 3)).values
        if m.sum() > 0:
            rows[c] = abs_df[m].mean()
    if rows:
        cty = pd.DataFrame(rows)
        cty_share = share(cty).round(1)
        print("\nFeature share % by country (horizon 3):")
        print(cty_share.to_string())
        cty_share.to_csv(RES / "shap_by_country_h3.csv")
        print("\nGrouped share % by country (horizon 3):")
        print(share(pd.DataFrame({
            c: g_all[((test["country"] == c) & (test["horizon"] == 3)).values].mean()
            for c in rows})).round(1).to_string())

    # 5. Local explanations
    print("\nLocal cases (values on log-ratio scale):")
    for c, y0, h in CASES:
        idx = np.where((test["country"] == c) & (test["year"] == y0)
                       & (test["horizon"] == h))[0]
        if len(idx) == 0:
            print(f"  {c} {y0} h={h}: not found")
            continue
        i = int(idx[0])
        pred = base + sv[i].sum()
        top = pd.Series(sv[i], index=X_COLS).sort_values(key=np.abs, ascending=False)[:3]
        print(f"  {c}, origin {y0}, h={h} (target {y0 + h}): "
              f"actual log-ratio {test['y'].iloc[i]:+.3f}, predicted {pred:+.3f}")
        print("    top contributions:", {k: round(float(v), 3) for k, v in top.items()})
        exp = shap.Explanation(values=sv[i], base_values=base,
                               data=X.iloc[i].values, feature_names=X_COLS)
        shap.plots.waterfall(exp, show=False)
        plt.title(f"{c}: origin {y0}, horizon {h}")
        plt.tight_layout()
        plt.savefig(FIG / f"shap_waterfall_{c.replace(' ', '_')}_{y0}_h{h}.png", dpi=150)
        plt.close()

    print(f"\nFigures saved in {FIG}")


if __name__ == "__main__":
    main()