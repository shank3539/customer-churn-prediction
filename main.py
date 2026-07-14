from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

from download_from_s3 import download_model_if_needed

# ----------------------------------
# Download artifacts from S3
# ----------------------------------

#download_model_if_needed()

# ----------------------------------
# Load artifacts
# ----------------------------------

MODEL_DIR = "mlmodel"

model = joblib.load(
    os.path.join(MODEL_DIR, "model.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

label_encoders = joblib.load(
    os.path.join(MODEL_DIR, "label_encoders.pkl")
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

    try:
        # Convert request to dataframe
        data = pd.DataFrame(
            [customer.model_dump()]
        )

        # -------------------------
        # Apply Label Encoding
        # -------------------------

        for column in label_encoders:

            if column in data.columns:

                value = data[column].iloc[0]

                # Prevent unknown categories
                if value not in label_encoders[column].classes_:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value '{value}' for column '{column}'. "
                               f"Allowed values: {list(label_encoders[column].classes_)}"
                    )

                data[column] = label_encoders[column].transform(
                    data[column]
                )

        # -------------------------
        # Match training feature order
        # -------------------------

        if hasattr(scaler, "feature_names_in_"):
            data = data[
                scaler.feature_names_in_
            ]

        # -------------------------
        # Scale complete dataset
        # -------------------------

        data_scaled = scaler.transform(
            data
        )

        # -------------------------
        # Prediction
        # -------------------------

        prediction = model.predict(
            data_scaled
        )[0]

        probability = model.predict_proba(
            data_scaled
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
                else "Customer Will Stay"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )