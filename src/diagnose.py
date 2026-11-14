from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
p = pd.read_csv(ROOT / "results" / "baseline_test_predictions.csv")
models = ["Naive", "Linear (Ridge)", "Random Forest", "XGBoost"]

def mae(d, m):
    return (d["target"] - d[m]).abs().mean()

print("MAE by target year (horizon 1):")
d1 = p[p["horizon"] == 1]
print(pd.DataFrame({m: d1.groupby("target_year").apply(lambda d: mae(d, m))
                    for m in models}).round(2).to_string())

print("\nMAE excluding target year 2020:")
ex = p[p["target_year"] != 2020]
print(pd.DataFrame({m: ex.groupby("horizon").apply(lambda d: mae(d, m))
                    for m in models}).round(2).to_string())

print("\nMedian absolute % error per row (all countries equal):")
for m in models:
    ape = (p["target"] - p[m]).abs() / p["target"].clip(lower=1e-6) * 100
    print(m, p.assign(ape=ape).groupby("horizon")["ape"].median().round(2).tolist())

print("\nMean signed error (bias), horizon 3 (positive = over-prediction):")
d3 = p[p["horizon"] == 3]
for m in models:
    print(m, round((d3[m] - d3["target"]).mean(), 2))