"""Streamlit front-end for the churn model.

Run from the project root:
    python -m streamlit run streamlit_app.py
It loads app/models/churn_pipeline.pkl directly, so the FastAPI server does not need to be running.
"""
import importlib.util
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent

# Load preprocessing.py by file path so another installed package named "app" cannot shadow ours.
_spec = importlib.util.spec_from_file_location("churn_preprocessing", ROOT / "app" / "services" / "preprocessing.py")
_pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pre)
ensure_column_order = _pre.ensure_column_order

MODEL_PATH = ROOT / "app" / "models" / "churn_pipeline.pkl"
STATES = ("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC "
          "ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC").split()

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_column_order(df.drop(columns=["churn"], errors="ignore").copy())
    probs = load_model().predict_proba(df)[:, 1]
    out = df.copy()
    out["churn_probability"] = probs
    out["prediction"] = ["Likely to Churn" if p >= 0.5 else "Unlikely to Churn" for p in probs]
    return out


st.title("📉 Customer Churn Predictor")

if not MODEL_PATH.exists():
    st.error(f"Model not found at {MODEL_PATH}. Train it first:\n\n"
             "`python -m app.services.train_model --data-path data/train.csv --output app/models/churn_pipeline.pkl`")
    st.stop()

tab_single, tab_batch = st.tabs(["Single customer", "Batch CSV"])

with tab_single:
    with st.form("single"):
        c1, c2, c3, c4 = st.columns(4)
        state = c1.selectbox("State", STATES, index=STATES.index("OH"))
        area_code = c2.selectbox("Area code", ["area_code_408", "area_code_415", "area_code_510"])
        account_length = c3.number_input("Account length (days)", 0, 1000, 103)
        cs_calls = c4.number_input("Customer service calls", 0, 50, 1)
        c1, c2, c3 = st.columns(3)
        intl_plan = c1.selectbox("International plan", ["no", "yes"])
        vm_plan = c2.selectbox("Voice mail plan", ["yes", "no"])
        vm_msgs = c3.number_input("Voice mail messages", 0, 200, 29)

        usage = {}
        for label, key, vals in [("Day", "day", (294.7, 95, 50.10)), ("Evening", "eve", (200.1, 105, 17.01)),
                                 ("Night", "night", (300.3, 127, 13.51)), ("International", "intl", (13.7, 6, 3.70))]:
            st.markdown(f"**{label} usage**")
            a, b, c = st.columns(3)
            usage[f"total_{key}_minutes"] = a.number_input("Minutes", 0.0, 5000.0, vals[0], key=f"{key}m")
            usage[f"total_{key}_calls"] = b.number_input("Calls", 0, 1000, vals[1], key=f"{key}c")
            usage[f"total_{key}_charge"] = c.number_input("Charge ($)", 0.0, 1000.0, vals[2], key=f"{key}ch")

        go = st.form_submit_button("Predict churn", type="primary")

    if go:
        record = {"state": state, "account_length": account_length, "area_code": area_code,
                  "international_plan": intl_plan, "voice_mail_plan": vm_plan,
                  "number_vmail_messages": vm_msgs, **usage, "number_customer_service_calls": cs_calls}
        res = predict(pd.DataFrame([record])).iloc[0]
        p = float(res["churn_probability"])
        m1, m2 = st.columns([1, 3])
        m1.metric("Churn probability", f"{p:.1%}")
        if p >= 0.5:
            m2.error(f"**{res['prediction']}**")
        else:
            m2.success(f"**{res['prediction']}**")
        st.progress(min(max(p, 0.0), 1.0))

with tab_batch:
    st.write("Upload a CSV with the same columns as `data/sample_upload.csv` (a `churn` column is ignored).")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        result = predict(data)
        st.dataframe(result, use_container_width=True)
        likely = (result["prediction"] == "Likely to Churn").sum()
        st.caption(f"{len(result)} rows — {likely} likely to churn")
        st.download_button("Download results", result.to_csv(index=False), "churn_predictions.csv", "text/csv")