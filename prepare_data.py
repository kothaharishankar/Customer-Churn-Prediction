"""Convert the Kaggle/BigML churn files into the format this project expects.

Put churn-bigml-80.csv and churn-bigml-20.csv in the project's data/ folder, then run from the project root:
    python prepare_data.py
It writes data/train.csv, data/test.csv and data/sample_upload.csv (the originals are left untouched).
"""
from pathlib import Path
import pandas as pd

DATA = Path("data")
RENAME = {"Customer service calls": "number_customer_service_calls", "Number vmail messages": "number_vmail_messages"}


def convert(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: RENAME.get(c, c.strip().lower().replace(" ", "_")))
    df["area_code"] = "area_code_" + df["area_code"].astype(str)
    for c in ["international_plan", "voice_mail_plan"]:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["churn"] = df["churn"].astype(str).str.strip().str.lower().map({"true": "yes", "false": "no"})
    assert df["churn"].notna().all(), "unexpected values in Churn column"
    return df


train = convert(pd.read_csv(DATA / "churn-bigml-80.csv"))
test = convert(pd.read_csv(DATA / "churn-bigml-20.csv"))
train.to_csv(DATA / "train.csv", index=False)
test.to_csv(DATA / "test.csv", index=False)
test.drop(columns="churn").head(20).to_csv(DATA / "sample_upload.csv", index=False)
print(f"train.csv: {len(train)} rows | test.csv: {len(test)} rows | churn rate {(train.churn == 'yes').mean():.1%}")