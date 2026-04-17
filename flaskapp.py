import os
import pandas as pd
import mlflow.pyfunc
import joblib
from fastapi import FastAPI

app = FastAPI()

# 1. GLOBAL INITIALIZATION
regression_model = None
classification_model = None

# -------------------------------
# MLFLOW & LOCAL FALLBACK LOADING
# -------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"Connecting to MLflow at: {MLFLOW_TRACKING_URI}")

try:
    print("Attempting to pull models from MLflow...")
    regression_model = mlflow.pyfunc.load_model("models:/VoyagePriceModel/latest")
    classification_model = mlflow.pyfunc.load_model("models:/VoyageGenderModel/latest")
    print("✅ MLflow models loaded.")
except Exception as e:
    print(f"⚠️ MLflow Load Failed: {e}")
    print("🔄 Falling back to local bundled models...")
    try:
        regression_model = joblib.load("models/regression_model.pkl")
        classification_model = joblib.load("models/classification_model.pkl")
        print("✅ Local fallback models loaded successfully.")
    except Exception as local_e:
        print(f"❌ CRITICAL ERROR: Fallback also failed: {local_e}")

# Load recommendation matrices
try:
    user_similarity = joblib.load("models/user_similarity.pkl")
    user_item = joblib.load("models/user_item.pkl")
    print("✅ Recommendation matrices loaded.")
except Exception as e:
    print(f"❌ Joblib Load Failed: {e}")
    user_similarity = None
    user_item = None

# -------------------------------
# COMMON FEATURE BUILDER
# -------------------------------
def build_full_features(data: dict):
    if "from_location" in data:
        data["from"] = data.pop("from_location")
    if "to_location" in data:
        data["to"] = data.pop("to_location")

    # Added missing features to match BOTH model schemas
    defaults = {
        "age": 30, "distance": 500, "price_y": 200, "time": 5,
        "days": 3, "place": "Delhi", "price_x": 150, "company": "Indigo",
        "gender": "male", "from": "Mumbai", "to": "Delhi",
        "flightType": "Economy", "agency": "MakeMyTrip",
        "return_month": 3, "travel_month": 3, "return_day": 24, "travel_day": 24,
        "total": 0  # Added to fix the classification schema error
    }

    for key, value in defaults.items():
        data.setdefault(key, value)

    return pd.DataFrame([data])

# -------------------------------
# ENDPOINTS
# -------------------------------

@app.post("/predict_price")
def predict_price(data: dict):
    if regression_model is None:
        return {"error": "Price model is not loaded."}
    try:
        df = build_full_features(data)
        pred = regression_model.predict(df)[0]
        return {"predicted_price": float(pred)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_gender")
def predict_gender(data: dict):
    if classification_model is None:
        return {"error": "Classification model is not loaded."}
    try:
        df = build_full_features(data)
        pred = classification_model.predict(df)[0]
        gender_map = {0: "female", 1: "male"}
        result = gender_map.get(pred, str(pred))
        return {"predicted_gender": result}
    except Exception as e:
        return {"error": str(e)}

@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    if user_item is None or user_similarity is None:
        return {"error": "Recommendation models not loaded."}
    try:
        if user_id not in user_item.index:
            return {"recommendations": [], "status": "User ID not found"}
        similar_users = user_similarity[user_id].sort_values(ascending=False)[1:6]
        recommendations = []
        for sim_user in similar_users.index:
            items = user_item.loc[sim_user]
            top_items = items[items > 0].index.tolist()
            recommendations.extend(top_items)
        recommendations = list(dict.fromkeys(recommendations))[:5]
        return {"user_id": user_id, "recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}