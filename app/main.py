from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from prophet.serialize import model_from_json
import json

from app.utils import preprocess_expense
from app.pipeline import detect_exact_duplicates, detect_near_duplicates, detect_outliers

# Initialize API
app = FastAPI(title="Expense Management ML API")

# Load models
rf_model = joblib.load("saved_models/rf_classifier.pkl")
isolation_model = joblib.load("saved_models/isolation_forest.pkl")
kmeans_model = joblib.load("saved_models/kmeans.pkl")
with open("saved_models/prophet_model.json", "r") as f:
    prophet_model = model_from_json(f.read())

class ExpenseRecord(BaseModel):
    Employee_ID: int
    Expense_Amount: float
    Currency: str
    Category: str
    Description: str
    Date: str
    Vendor: str
    Department: str
    Location: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_expense(exp: ExpenseRecord):
    """Classify expense as Approved/Rejected"""
    df = pd.DataFrame([exp.dict()])
    X = preprocess_expense(df)
    pred = rf_model.predict(X)[0]
    return {"prediction": str(pred)}

@app.post("/detect-duplicates")
def detect_duplicates(records: list[ExpenseRecord]):
    df = pd.DataFrame([r.dict() for r in records])
    exact_dupes = detect_exact_duplicates(df)
    near_dupes = detect_near_duplicates(df)
    return {"exact_duplicates": exact_dupes, "near_duplicates": near_dupes}

@app.post("/detect-outliers")
def detect_outlier(records: list[ExpenseRecord]):
    df = pd.DataFrame([r.dict() for r in records])
    flags = detect_outliers(df, isolation_model, kmeans_model)
    return {"outlier_flags": flags}

@app.get("/forecast")
def forecast(n_periods: int = 12):
    future = prophet_model.make_future_dataframe(periods=n_periods, freq="M")
    forecast = prophet_model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(n_periods).to_dict(orient="records")
