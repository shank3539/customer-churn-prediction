# Customer Churn Prediction API

## Project Overview

Customer churn is one of the biggest challenges faced by telecom companies. This project predicts whether a customer is likely to leave the service based on customer information and subscription details.

The project uses Machine Learning for prediction and exposes the model through a FastAPI REST API. The application is containerized using Docker and supports cloud deployment.

---

## Features

- Customer churn prediction using Machine Learning
- REST API using FastAPI
- Interactive Swagger API documentation
- Dockerized deployment
- AWS S3 support for model storage
- Ready for deployment on Render, AWS, or other cloud platforms

---

## Tech Stack

### Machine Learning
- Scikit-Learn
- XGBoost
- Pandas
- NumPy
- Joblib

### Backend
- FastAPI
- Uvicorn

### DevOps
- Docker
- AWS S3
- GitHub

---

## Project Structure

```text
customer-churn-prediction/
│
├── app/
├── artifacts/
├── data/
├── mlmodel/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
│
├── notebooks/
├── SRC/
│
├── main.py
├── download_from_s3.py
├── config.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .env
```

---

## Dataset Features

The model uses the following customer information:

- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shank3539/customer-churn-prediction.git
cd customer-churn-prediction
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Locally

Start FastAPI server:

```bash
uvicorn main:app --reload
```

Application:

```text
http://localhost:8000
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

## Docker Setup

Build Docker image:

```bash
docker build -t churn-app .
```

Run Docker container:

```bash
docker run -p 8000:8000 churn-app
```

Swagger URL:

```text
http://localhost:8000/docs
```

---

## API Endpoint

### POST `/predict`

Sample Request:

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 844.20
}
```

Sample Response:

```json
{
  "prediction": 0,
  "churn_probability": 0.1842,
  "result": "Customer Will Stay"
}
```

---

## Environment Variables

Create a `.env` file:

```env
AWS_ACCESS_KEY=your_access_key
AWS_SECRET_KEY=your_secret_key
AWS_REGION=your_region
BUCKET_NAME=your_bucket_name
```

---

## Model Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Label Encoding
5. Feature Scaling
6. Model Training
7. Model Evaluation
8. Model Deployment
9. Prediction API

---

## Deployment Options

- AWS Elastic Beanstalk
- AWS ECS
- Render
- Railway
- Docker

---

## Author

**Shashank Rai**

Data Science Student  
Machine Learning Enthusiast

GitHub:
:contentReference[oaicite:0]{index=0}

Repository:
:contentReference[oaicite:1]{index=1}