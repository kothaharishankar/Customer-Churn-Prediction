import os
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

# Ensure a minimal pipeline exists for tests so imports/health check succeed
MODEL_PATH = "app/models/churn_pipeline.pkl"
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # Create a minimal classifier that supports predict_proba
    # The API expects 19 features (EXPECTED_COLUMNS in preprocessing), so fit on dummy data
    clf = Pipeline([("clf", DummyClassifier(strategy="prior"))])
    X = np.zeros((2, 19))
    y = np.array([0, 1])
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_single():
    payload = {
        "records": [{
            "state": "OH",
            "account_length": 103,
            "area_code": "area_code_408",
            "international_plan": "no",
            "voice_mail_plan": "yes",
            "number_vmail_messages": 29,
            "total_day_minutes": 294.7,
            "total_day_calls": 95,
            "total_day_charge": 50.10,
            "total_eve_minutes": 200.1,
            "total_eve_calls": 105,
            "total_eve_charge": 17.01,
            "total_night_minutes": 300.3,
            "total_night_calls": 127,
            "total_night_charge": 13.51,
            "total_intl_minutes": 13.7,
            "total_intl_calls": 6,
            "total_intl_charge": 3.70,
            "number_customer_service_calls": 1
        }]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "results" in response.json()

def test_predict_invalid_input():
    response = client.post("/predict", json={"records": []})
    assert response.status_code == 400
