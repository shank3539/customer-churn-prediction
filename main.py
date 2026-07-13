from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

from download_from_s3 import download_model_if_needed

# ----------------------------------
# Download from S3 if missing
# ----------------------------------

download_model_if_needed()

# ----------------------------------
# Load artifacts
# ----------------------------------

MODEL_DIR = "registered_model/mlmodel"

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "model.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

label_encoders = joblib.load(
    os.path.join(
        MODEL_DIR,
        "label_encoders.pkl"
    )
)

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn API Running"
    }


@app.post("/predict")
def predict(customer: Customer):

    data = pd.DataFrame(
        [customer.model_dump()]
    )

    # -------------------------
    # Apply Label Encoding
    # -------------------------

    for column in label_encoders:

        if column in data.columns:
            data[column] = label_encoders[
                column
            ].transform(
                data[column]
            )

    # -------------------------
    # Scale Numeric Features
    # -------------------------

    numeric_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    data[numeric_cols] = scaler.transform(
        data[numeric_cols]
    )

    prediction = model.predict(
        data
    )[0]

    probability = model.predict_proba(
        data
    )[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": round(
            float(probability),
            4
        ),
        "result": (
            "Customer Will Churn"
            if prediction == 1
            else
            "Customer Will Stay"
        )
    }