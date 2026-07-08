from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="Customer Churn Prediction API"
)

BASE_DIR = Path(__file__).resolve().parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"

model = joblib.load(
    ARTIFACTS_DIR / "model.pkl"
)

preprocessor = joblib.load(
    ARTIFACTS_DIR / "preprocessor.pkl"
)   

class Customer(BaseModel):
    gender:str
    SeniorCitizen:int
    Partner:str
    Dependents:str
    tenure:int
    PhoneService:str
    MultipleLines:str
    InternetService:str
    OnlineSecurity:str
    OnlineBackup:str
    DeviceProtection:str
    TechSupport:str
    StreamingTV:str
    StreamingMovies:str
    Contract:str
    PaperlessBilling:str
    PaymentMethod:str
    MonthlyCharges:float
    TotalCharges:float

@app.get("/")
def home():
    return {
        "message":"Customer Churn API Running"
    }

@app.post("/predict")
def predict(customer: Customer):

    data = pd.DataFrame(
        [customer.dict()]
    )

    processed_data = preprocessor.transform(
        data
    )

    prediction = model.predict(
        processed_data
    )[0]

    probability = model.predict_proba(
        processed_data
    )[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
        "result":
            "Customer Will Churn"
            if prediction == 1
            else
            "Customer Will Stay"
    }

