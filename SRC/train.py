import os
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd

from xgboost import XGBClassifier
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==========================================================
# Configuration
# ==========================================================

EXPERIMENT_NAME = "Telco-Customer-Churn-XGBoost"
REGISTERED_MODEL_NAME = "CustomerChurnModel"

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("Telco-Customer-Churn.csv")

# Remove customer ID
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges column
df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# ==========================================================
# Encode Categorical Columns
# ==========================================================

label_encoders = {}

for column in df.columns:
    if df[column].dtype == "object":
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        label_encoders[column] = encoder

# Save encoders
joblib.dump(
    label_encoders,
    os.path.join(
        ARTIFACTS_DIR,
        "label_encoders.pkl"
    )
)

# ==========================================================
# Features and Target
# ==========================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ==========================================================
# Train/Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# Scaling
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    os.path.join(
        ARTIFACTS_DIR,
        "scaler.pkl"
    )
)

# ==========================================================
# Train Multiple Models
# ==========================================================

depths = [2, 4, 6, 8, 10]

best_model = None
best_f1 = 0
best_depth = None

for depth in depths:

    with mlflow.start_run(
        run_name=f"XGBoost_Depth_{depth}"
    ):

        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )

        # Training
        model.fit(
            X_train,
            y_train
        )

        # Prediction
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        # Console Output
        print("\n" + "=" * 50)
        print(f"Depth      : {depth}")
        print(f"Accuracy   : {accuracy:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1 Score   : {f1:.4f}")
        print(f"ROC AUC    : {roc_auc:.4f}")

        # Log Parameters
        mlflow.log_params({
            "model_type": "XGBoost",
            "max_depth": depth,
            "n_estimators": 200,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        })

        # Log Metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })

        # Log Model
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model"
        )

        # Track best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_depth = depth

# ==========================================================
# Save Best Model
# ==========================================================

best_model_path = os.path.join(
    ARTIFACTS_DIR,
    "model.pkl"
)

joblib.dump(
    best_model,
    best_model_path
)

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)
print(f"Best Depth    : {best_depth}")
print(f"Best F1 Score : {best_f1:.4f}")

# ==========================================================
# Register Best Model
# ==========================================================

with mlflow.start_run(
    run_name="Best_Model_Registration"
):

    model_info = mlflow.xgboost.log_model(
        xgb_model=best_model,
        name="registered_model"
    )

    mlflow.log_artifact(best_model_path)
    mlflow.log_artifact(
        os.path.join(
            ARTIFACTS_DIR,
            "scaler.pkl"
        )
    )

    result = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=REGISTERED_MODEL_NAME
    )

    print(
        f"\nRegistered Model Name: {REGISTERED_MODEL_NAME}"
    )

    print(
        f"Registered Version: {result.version}"
    )

print("\nTraining Completed Successfully")