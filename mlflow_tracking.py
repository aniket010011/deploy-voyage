import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier

# Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from mlflow.models.signature import infer_signature

# -----------------------------
# MLflow Setup
# -----------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Voyage Advanced Tracking V2")

mlflow.sklearn.autolog()

# -----------------------------
# LOAD DATA
# -----------------------------
users = pd.read_csv("users.csv")
hotels = pd.read_csv("hotels.csv")
flights = pd.read_csv("flights.csv")

df = flights.merge(hotels, on=["travelCode", "userCode"])
df = df.merge(users, left_on="userCode", right_on="code")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df['date_x'] = pd.to_datetime(df['date_x'])
df['date_y'] = pd.to_datetime(df['date_y'])

# Extract useful features
df["travel_month"] = df["date_x"].dt.month
df["travel_day"] = df["date_x"].dt.day
df["return_month"] = df["date_y"].dt.month
df["return_day"] = df["date_y"].dt.day

# 🚨 DROP datetime columns (CRITICAL FIX)
df = df.drop(columns=["date_x", "date_y"])

# Remove invalid gender
df = df[df["gender"] != "none"]

# =====================================================
# 🔥 REGRESSION
# =====================================================
print("\n🚀 Running Regression Models...")

X_reg = df.drop(columns=[
    "total", "travelCode", "userCode", "code", "name_x", "name_y"
])
y_reg = df["total"]

categorical_cols = X_reg.select_dtypes(include=["object"]).columns
numeric_cols = X_reg.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

models_reg = {
    "LinearRegression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]),
    "XGBRegressor": Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor())
    ])
}

best_rmse = float("inf")
best_model_reg = None
best_model_name_reg = ""

mlflow.set_experiment("Voyage Regression Model")

for name, model in models_reg.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train_reg, y_train_reg)
        preds = model.predict(X_test_reg)

        rmse = np.sqrt(mean_squared_error(y_test_reg, preds))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("model_name", name)

        print(f"{name} RMSE: {rmse}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_reg = model
            best_model_name_reg = name

# Register best regression model
with mlflow.start_run(run_name="Best_Regression_Model"):
    signature = infer_signature(X_train_reg, y_train_reg)

    mlflow.sklearn.log_model(
        best_model_reg,
        "best_regression_model",
        signature=signature,
        registered_model_name="VoyagePriceModel"
    )

print(f"\n✅ Best Regression Model: {best_model_name_reg} | RMSE: {best_rmse}")


# =====================================================
# 🔥 CLASSIFICATION
# =====================================================
print("\n🚀 Running Classification Models...")

X_clf = df.drop(columns=[
    "gender", "travelCode", "userCode", "code", "name_x", "name_y"
])

le = LabelEncoder()
y_clf = le.fit_transform(df["gender"])

categorical_cols_clf = X_clf.select_dtypes(include=["object"]).columns
numeric_cols_clf = X_clf.select_dtypes(exclude=["object"]).columns

preprocessor_clf = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols_clf),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols_clf)
])

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

models_clf = {
    "LogisticRegression": Pipeline([
        ("preprocessor", preprocessor_clf),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "XGBClassifier": Pipeline([
        ("preprocessor", preprocessor_clf),
        ("model", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])
}

best_acc = 0
best_model_clf = None
best_model_name_clf = ""

mlflow.set_experiment("Voyage Classification Model")

for name, model in models_clf.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train_clf, y_train_clf)
        preds = model.predict(X_test_clf)

        acc = accuracy_score(y_test_clf, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("model_name", name)

        print(f"{name} Accuracy: {acc}")

        if acc > best_acc:
            best_acc = acc
            best_model_clf = model
            best_model_name_clf = name

# Register best classification model
with mlflow.start_run(run_name="Best_Classification_Model"):
    signature = infer_signature(X_train_clf, y_train_clf)

    mlflow.sklearn.log_model(
        best_model_clf,
        "best_classification_model",
        signature=signature,
        registered_model_name="VoyageGenderModel"
    )

print(f"\n✅ Best Classification Model: {best_model_name_clf} | Accuracy: {best_acc}")