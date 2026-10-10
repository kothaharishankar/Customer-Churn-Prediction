from typing import List, Tuple
import pandas as pd

CATEGORICAL = ["state","area_code","international_plan","voice_mail_plan"]
TARGET = "churn"

# For training scripts that start from a CSV with target
def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = None
    if TARGET in df.columns:
        y = (df[TARGET].astype(str).str.lower() == "yes").astype(int)
        df = df.drop(columns=[TARGET])
    return df, y

EXPECTED_COLUMNS = [
    "state","account_length","area_code","international_plan","voice_mail_plan",
    "number_vmail_messages","total_day_minutes","total_day_calls","total_day_charge",
    "total_eve_minutes","total_eve_calls","total_eve_charge",
    "total_night_minutes","total_night_calls","total_night_charge",
    "total_intl_minutes","total_intl_calls","total_intl_charge",
    "number_customer_service_calls"
]

def ensure_column_order(df: pd.DataFrame) -> pd.DataFrame:
    # Add any missing expected columns with safe defaults
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    # Drop unexpected columns (keep a safe schema for the pipeline)
    df = df[EXPECTED_COLUMNS]
    return df
