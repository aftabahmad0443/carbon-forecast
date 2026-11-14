from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RES = Path(__file__).resolve().parent.parent / "results"

st.set_page_config(page_title="CO2 forecast with uncertainty", layout="wide")


@st.cache_data
def load():
    iv = pd.read_csv(RES / "conformal_test_intervals.csv")
    shap_share = pd.read_csv(RES / "shap_share_no_horizon.csv", index_col=0)
    return iv, shap_share


try:
    iv, shap_share = load()
except FileNotFoundError as e:
    st.error(
        f"Missing results file: {e}. "
        "Run src/conformal.py and src/shap_check.py first, and make sure "
        "the results/ folder is committed to GitHub."
    )
    st.stop()

st.title("Explainable, uncertainty-aware CO2 forecasting")
st.caption(
    "Random Forest forecasts with split-conformal 90% intervals, "
    "evaluated on target years 2016-2022."
)

countries = sorted(iv["country"].unique())
default = countries.index("Pakistan") if "Pakistan" in countries else 0
country = st.sidebar.selectbox("Country", countries, index=default)
horizon = st.sidebar.selectbox("Forecast horizon (years ahead)", [1, 3, 5])
kind = st.sidebar.radio("Interval type", ["Static", "Rolling"])
lo_col, hi_col = (
    ("lo_static", "hi_static") if kind == "Static" else ("lo_roll", "hi_roll")
)

d = (
    iv[(iv["country"] == country) & (iv["horizon"] == horizon)]
    .sort_values("target_year")
)
d = d.assign(inside=(d["target"] >= d[lo_col]) & (d["target"] <= d[hi_col]))

if d.empty:
    st.warning("No forecasts available for this country and horizon.")
    st.stop()

left, right = st.columns([3, 2])

with left:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["target_year"], y=d[hi_col],
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["target_year"], y=d[lo_col], fill="tonexty",
            line=dict(width=0), name="90% interval",
            fillcolor="rgba(70,130,180,0.25)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["target_year"], y=d["pred"],
            mode="lines+markers", name="Forecast",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["target_year"], y=d["target"],
            mode="lines+markers", name="Actual", line=dict(dash="dot"),
        )
    )
    fig.update_layout(
        xaxis_title="Target year",
        yaxis_title="CO2 (Mt)",
        title=f"{country}: {horizon}-year-ahead forecast",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch")

    n, k = len(d), int(d["inside"].sum())
    st.write(
        f"Actual values inside the interval: **{k} of {n}** for this country "
        f"(too few points to judge calibration; the target is 90% overall)."
    )

with right:
    st.subheader("What the model relies on")
    col = f"h={horizon}"
    if col in shap_share.columns:
        s = shap_share[col].sort_values()
        bar = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h"))
        bar.update_layout(xaxis_title="Share of mean |SHAP| (%)", height=380)
        st.plotly_chart(bar, width="stretch")
        st.caption(
            "SHAP describes what the model uses, not what causes emissions. "
            "Importances are global (all test countries), not specific to the "
            "selected one."
        )
    else:
        st.info(f"No SHAP data found for {col}.")

st.subheader("Forecast table")
st.dataframe(
    d[["target_year", "pred", lo_col, hi_col, "target", "inside"]].round(2),
    hide_index=True,
)

with st.expander("Limitations"):
    st.markdown(
        "- Overall test coverage was about 88.9 / 88.3 / 86.5% (h=1/3/5) "
        "against a 90% target.\n"
        "- Coverage dropped to about 83% for target year 2020 (COVID shock).\n"
        "- Under held-out-continent tests, coverage fell further "
        "(about 80-86% pooled).\n"
        "- Intervals are wide: median width is roughly 31% / 53% / 66% of "
        "the forecast.\n"
        "- On annual data, the model beats a naive 'no change' forecast "
        "clearly only at 5 years."
    )