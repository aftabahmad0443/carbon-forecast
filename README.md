# Explainable and Uncertainty-Aware AI for Carbon Emission Prediction

Annual country-level CO2 forecasting (1, 3 and 5 years ahead) with SHAP explanations
and conformal prediction intervals, evaluated under temporal and geographical shift.

**Status:** research in progress. The research gap is provisional until the literature review is complete.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.14 on Windows. Run all scripts from the project root.

## Data

Download `owid-co2-data.csv` from https://github.com/owid/co2-data and save it as
`data/raw/owid-co2-data.csv`. Raw and processed data are not committed (see `.gitignore`).

## Pipeline (run in this order)

| Step | Command | Output |
|---|---|---|
| 1. Data and features | `python src\data_prep.py` | `data/processed/*.csv` |
| 2. Baselines | `python src\baselines.py` | `results/baseline_metrics.csv` |
| 3. Tuning, drift baseline | `python src\tune.py` | `results/tuned_metrics.csv` |
| 4. Conformal intervals | `python src\conformal.py` | `results/conformal_*.csv` |
| 5. SHAP | `python src\explain.py`, `python src\shap_check.py` | `results/shap_*.csv`, figures |
| 6. Unseen countries | `python src\geographic.py` | `results/geographic_metrics.csv` |
| 7. Leave-one-continent-out | `python src\region.py` (needs `pycountry-convert`) | `results/region_metrics.csv` |
| 8. LSTM | `python src\lstm.py` | `results/lstm_metrics.csv` |
| 9. Bootstrap CIs | `python src\bootstrap.py` | `results/bootstrap_ci.csv` |
| 10. Demo | `streamlit run app\streamlit_app.py` | local web app |

## Design

- Target: CO2 (Mt) at t+h, modelled as log(target / current CO2).
- Split by target year: train up to 2010, calibration 2011-2015, test 2016-2022.
- 153 countries with complete data for 1990-2022. Features use past information only.
- Models: Naive (no change), Ridge, Random Forest, XGBoost, LSTM (3-seed ensemble).
- Conformal: split conformal on the log-ratio scale, per horizon, target coverage 90%.

## Main results (test period, 2016-2022)

Mean absolute log error (lower is better):

| Horizon | Naive | RF | LSTM |
|---|---|---|---|
| 1 year | 0.075 | 0.077 | 0.073 |
| 3 years | 0.134 | 0.130 | 0.130 |
| 5 years | 0.196 | 0.175 | 0.172 |

Conformal coverage (target 90%): RF 88.9 / 88.3 / 86.5%, LSTM 89.6 / 88.6 / 87.1%.

Findings so far:
- Complex models beat the naive forecast clearly only at 5 years (country-bootstrap 95% CIs exclude 0).
  RF and LSTM are not clearly different from each other, except a small LSTM edge at 1 year.
- Coverage is slightly below target and weakest in 2020 (about 83%).
- Held-out-continent tests reduce coverage further (pooled 86 / 83 / 80% at h=1/3/5), worst for Africa and Europe.
- SHAP (horizon excluded): scale features about 56%, economy/energy 22%, momentum 22%.

## Limitations

- Bootstrap resamples countries only and ignores common year shocks, so intervals are likely too narrow.
- Single architecture and limited tuning for the LSTM; no Transformer was run.
- SHAP describes model behaviour, not causes of emissions.
- Per-country results (e.g. Pakistan) rest on 7 test observations per horizon.
- Annual data gives few training sequences per country.