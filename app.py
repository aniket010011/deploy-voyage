import streamlit as st
import pandas as pd
import mlflow.pyfunc
import joblib

# 1. Connect directly to the local MLflow database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 2. Cache models so they only load once
@st.cache_resource
def load_models():
    try:
        reg_model = mlflow.pyfunc.load_model("models:/VoyagePriceModel/latest")
        clf_model = mlflow.pyfunc.load_model("models:/VoyageGenderModel/latest")
        user_sim = joblib.load("models/user_similarity.pkl")
        user_itm = joblib.load("models/user_item.pkl")
        return reg_model, clf_model, user_sim, user_itm
    except Exception as e:
        st.error(f"Failed to load models. Ensure mlflow_tracking.py ran successfully. Error: {e}")
        return None, None, None, None

regression_model, classification_model, user_similarity, user_item = load_models()

# 3. Feature Builder (Matches your training logic)
def build_full_features(data: dict):
    # Handle naming mismatches
    if "from_location" in data:
        data["from"] = data.pop("from_location")
    if "to_location" in data:
        data["to"] = data.pop("to_location")

    # Handle Date Features
    if "date_x" in data:
        date_x = pd.to_datetime(data["date_x"])
        data["travel_month"] = date_x.month
        data["travel_day"] = date_x.day
        del data["date_x"]
    else:
        data["travel_month"] = 1
        data["travel_day"] = 15

    if "date_y" in data:
        date_y = pd.to_datetime(data["date_y"])
        data["return_month"] = date_y.month
        data["return_day"] = date_y.day
        del data["date_y"]
    else:
        data["return_month"] = 1
        data["return_day"] = 20

    defaults = {
        "age": 30, "distance": 500, "price_y": 200, "time": 5,
        "days": 3, "place": "Delhi", "price_x": 150, "company": "Indigo",
        "gender": "male", "from": "Mumbai", "to": "Delhi",
        "flightType": "Economy", "agency": "MakeMyTrip"
    }

    for key, value in defaults.items():
        data.setdefault(key, value)
        
    cols_to_drop = ["total", "travelCode", "userCode", "code", "name_x", "name_y"]
    for col in cols_to_drop:
        if col in data:
            del data[col]

    return pd.DataFrame([data])

# 4. Streamlit UI
st.title("Voyage Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(["Price Prediction", "Gender Classification", "Recommendations"])

with tab1:
    st.header("Predict Travel Price")
    with st.form("price_form"):
        distance = st.number_input("Distance", value=500)
        days = st.number_input("Days", value=3)
        company = st.selectbox("Airline", ["Indigo", "AirIndia", "Vistara"])
        flightType = st.selectbox("Class", ["Economy", "Premium Economy", "Business"])
        submit_price = st.form_submit_button("Predict")
        
        if submit_price and regression_model:
            input_data = {"distance": distance, "days": days, "company": company, "flightType": flightType}
            df = build_full_features(input_data)
            pred = regression_model.predict(df)[0]
            st.success(f"Predicted Total Price: ${pred:.2f}")

with tab2:
    st.header("Classify Gender")
    with st.form("gender_form"):
        distance_g = st.number_input("Distance", value=500, key="dist_g")
        days_g = st.number_input("Days", value=3, key="days_g")
        company_g = st.selectbox("Airline", ["Indigo", "AirIndia", "Vistara"], key="comp_g")
        submit_gender = st.form_submit_button("Classify")
        
        if submit_gender and classification_model:
            input_data = {"distance": distance_g, "days": days_g, "company": company_g}
            df = build_full_features(input_data)
            pred = classification_model.predict(df)[0]
            label = "Male" if pred == 1 else "Female"
            st.info(f"Predicted Gender: {label}")

with tab3:
    st.header("Get Recommendations")
    user_id = st.number_input("Enter User ID", min_value=1, value=1, step=1)
    if st.button("Recommend"):
        if user_similarity is not None and user_id in user_item.index:
            similar_users = user_similarity[user_id].sort_values(ascending=False)[1:6]
            recommendations = []
            for sim_user in similar_users.index:
                items = user_item.loc[sim_user]
                top_items = items[items > 0].index.tolist()
                recommendations.extend(top_items)
            
            recommendations = list(dict.fromkeys(recommendations))[:5]
            st.write("Top Recommended Items:")
            for item in recommendations:
                st.write(f"- {item}")
        else:
            st.warning("User ID not found or models not loaded.")
