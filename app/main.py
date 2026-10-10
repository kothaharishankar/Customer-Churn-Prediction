from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.routers import predict, upload
from pathlib import Path
import os

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

app.include_router(predict.router, prefix="/predict", tags=["predict"])
app.include_router(upload.router,  prefix="/upload",  tags=["upload"])

STATIC_DIR = Path(__file__).parent / "static"

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
def health():
    model_path = "app/models/churn_pipeline.pkl"
    model_loaded = os.path.exists(model_path)
    return {"status": "healthy" if model_loaded else "unhealthy", "model_loaded": model_loaded}