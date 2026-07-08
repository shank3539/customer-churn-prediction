import joblib
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"

model = joblib.load(
    ARTIFACTS_DIR / "model.pkl"
)

preprocessor = joblib.load(
    ARTIFACTS_DIR / "preprocessor.pkl"
)

sample = {
    "gender":"Male",
    "SeniorCitizen":0,
    "Partner":"No",
    "Dependents":"No",
    "tenure":1,
    "PhoneService":"No",
    "MultipleLines":"No phone service",
    "InternetService":"DSL",
    "OnlineSecurity":"No",
    "OnlineBackup":"Yes",
    "DeviceProtection":"No",
    "TechSupport":"No",
    "StreamingTV":"No",
    "StreamingMovies":"No",
    "Contract":"Month-to-month",
    "PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check",
    "MonthlyCharges":29.85,
    "TotalCharges":29.85
}

sample_df = pd.DataFrame([sample])

sample_processed = preprocessor.transform(
    sample_df
)

prediction = model.predict(
    sample_processed
)[0]

probability = model.predict_proba(
    sample_processed
)[0][1]

if prediction == 1:
    print("Customer will churn")
else:
    print("Customer will stay")

print("Probability:", probability)

