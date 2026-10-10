from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.schemas import UploadResponse, PredictResponseItem
from app.services.preprocessing import ensure_column_order
import pandas as pd, joblib, os, io

router = APIRouter()

MODEL_PATH = os.getenv("MODEL_PATH", "app/models/churn_pipeline.pkl")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Train it first.")

pipeline = joblib.load(MODEL_PATH)

@router.post("", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported")
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    if "churn" in df.columns:
        df = df.drop(columns=["churn"])  # ignore target if provided
    df = ensure_column_order(df)
    probs = pipeline.predict_proba(df)[:, 1]
    preds = ["Likely to Churn" if p >= 0.5 else "Unlikely to Churn" for p in probs]
    items = [PredictResponseItem(churn_probability=float(p), prediction=pred) for p, pred in zip(probs, preds)]
    return UploadResponse(rows=len(df), results=items)
