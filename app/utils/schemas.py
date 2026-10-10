from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class CustomerRecord(BaseModel):
    state: str
    account_length: int
    area_code: str
    international_plan: Literal["yes","no"]
    voice_mail_plan: Literal["yes","no"]
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: int
    total_eve_charge: float
    total_night_minutes: float
    total_night_calls: int
    total_night_charge: float
    total_intl_minutes: float
    total_intl_calls: int
    total_intl_charge: float
    number_customer_service_calls: int

class PredictRequest(BaseModel):
    records: List[CustomerRecord] = Field(default_factory=list)

class PredictResponseItem(BaseModel):
    churn_probability: float
    prediction: Literal["Likely to Churn","Unlikely to Churn"]

class PredictResponse(BaseModel):
    results: List[PredictResponseItem]

class UploadResponse(BaseModel):
    rows: int
    results: List[PredictResponseItem]
