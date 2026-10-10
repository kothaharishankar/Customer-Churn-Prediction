from fastapi import APIRouter, HTTPException
from app.utils.schemas import PredictRequest, PredictResponse, PredictResponseItem
from app.services.preprocessing import ensure_column_order
import pandas as pd, joblib, os

router = APIRouter()

MODEL_PATH = os.getenv("MODEL_PATH", "app/models/churn_pipeline.pkl")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Train it first.")

pipeline = joblib.load(MODEL_PATH)

@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.records:
        raise HTTPException(status_code=400, detail="No records provided")
    df = pd.DataFrame([r.model_dump() for r in req.records])
    df = ensure_column_order(df)
    probs = pipeline.predict_proba(df)[:, 1]
    preds = ["Likely to Churn" if p >= 0.5 else "Unlikely to Churn" for p in probs]
    items = [PredictResponseItem(churn_probability=float(p), prediction=pred) for p, pred in zip(probs, preds)]
    return PredictResponse(results=items)
