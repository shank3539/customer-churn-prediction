import os
import boto3

from config import (
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    AWS_REGION,
    BUCKET_NAME
)

LOCAL_DIR = "registered_model/mlmodel"

FILES = [
    "registered_model/mlmodel/model.pkl",
    "registered_model/mlmodel/scaler.pkl",
    "registered_model/mlmodel/label_encoders.pkl"
]

def download_model_if_needed():

    model_path = os.path.join(
        LOCAL_DIR,
        "model.pkl"
    )

    if os.path.exists(model_path):
        print("Model already exists locally.")
        return

    print("Downloading artifacts from S3...")

    os.makedirs(
        LOCAL_DIR,
        exist_ok=True
    )

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

    for file in FILES:

        local_path = os.path.join(
            LOCAL_DIR,
            os.path.basename(file)
        )

        try:
            print(f"Downloading {file}")

            s3.download_file(
                BUCKET_NAME,
                file,
                local_path
            )

            print(f"Downloaded {file}")

        except Exception as e:
            print(f"Failed downloading {file}")
            print(e)
            raise

    print("Artifacts downloaded successfully.")