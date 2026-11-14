import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from utils import SEED, set_seed

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

FEATURES = [
    "co2_lag1", "co2_lag2", "co2_lag3", "co2_growth", "co2_roll3",
    "log_gdp_pc", "log_population", "energy_per_capita", "co2_per_capita",
]


def load(name):
    return pd.read_csv(PROC / f"{name}.csv")


def add_log_features(df):
    """Scale-free inputs: log of co2 levels relative to current co2."""
    df = df.copy()
    eps = 1e-6
    df["lag1_ratio"] = np.log((df["co2_lag1"] + eps) / (df["co2"] + eps))
    df["lag2_ratio"] = np.log((df["co2_lag2"] + eps) / (df["co2"] + eps))
    df["lag3_ratio"] = np.log((df["co2_lag3"] + eps) / (df["co2"] + eps))
    df["roll3_ratio"] = np.log((df["co2_roll3"] + eps) / (df["co2"] + eps))
    df["log_co2"] = np.log(df["co2"] + eps)
    df["log_co2_pc"] = np.log(df["co2_per_capita"] + eps)
    df["log_energy_pc"] = np.log(df["energy_per_capita"] + 1)
    df["co2_growth_c"] = df["co2_growth"].clip(-1, 1)
    df["y"] = np.log((df["target"] + eps) / (df["co2"] + eps))
    return df


X_COLS = [
    "lag1_ratio", "lag2_ratio", "lag3_ratio", "roll3_ratio", "co2_growth_c",
    "log_co2", "log_co2_pc", "log_gdp_pc", "log_population", "log_energy_pc",
    "horizon",
]


def to_level(df, y_log_pred):
    return (df["co2"].values + 1e-6) * np.exp(y_log_pred) - 1e-6


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1e-6)) * 100),
    }


def main():
    set_seed()
    RES.mkdir(exist_ok=True)
    train, calib, test = (add_log_features(load(n)) for n in ("train", "calib", "test"))

    models = {
        "Linear (Ridge)": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, n_jobs=-1),
    }

    rows = []
    preds = test[["country", "year", "horizon", "target_year", "target", "co2"]].copy()

    # Naive baseline: predict that emissions stay at last known value
    preds["Naive"] = test["co2"].values

    scaler = StandardScaler().fit(train[X_COLS])
    for name, model in models.items():
        if name.startswith("Linear"):
            model.fit(scaler.transform(train[X_COLS]), train["y"])
            yhat = model.predict(scaler.transform(test[X_COLS]))
        else:
            model.fit(train[X_COLS], train["y"])
            yhat = model.predict(test[X_COLS])
        preds[name] = to_level(test, yhat)

    for h in sorted(test["horizon"].unique()):
        sub = preds[preds["horizon"] == h]
        for name in ["Naive"] + list(models):
            m = metrics(sub["target"].values, sub[name].values)
            rows.append({"horizon": h, "model": name, **m})

    out = pd.DataFrame(rows).round(3)
    print(out.to_string(index=False))
    out.to_csv(RES / "baseline_metrics.csv", index=False)
    preds.to_csv(RES / "baseline_test_predictions.csv", index=False)


if __name__ == "__main__":
    main()