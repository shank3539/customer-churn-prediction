import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("clean_churn.csv")

df.drop("customerID", axis=1, inplace=True)

print(df.head())

df["Churn"] = df["Churn"].map({
    "Yes":1,
    "No":0
})

print(df["Churn"].value_counts())

X = df.drop("Churn", axis=1)

y = df["Churn"]

print(X.shape)

print(y.shape)

categorical_columns = X.select_dtypes(include=["object"]).columns

print(categorical_columns)

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns

print(numerical_columns)

numeric_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numerical_columns
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_columns
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("Train:", X_train_processed.shape)

print("Test :", X_test_processed.shape)

joblib.dump(
    preprocessor,
    "artifacts/preprocessor.pkl"
)

print("Preprocessor saved successfully!")

