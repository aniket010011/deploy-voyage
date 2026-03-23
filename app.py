import streamlit as st
import requests

API_URL = "http://a6720fa00f3da44019c760cf5cdd2607-798674393.eu-north-1.elb.amazonaws.com"

st.set_page_config(page_title="Voyage Analytics", layout="wide")

st.sidebar.title("✈️ Voyage Analytics")
feature = st.sidebar.radio(
    "Select Feature",
    ["Price Prediction", "Gender Prediction", "Recommendation"]
)

# -------------------------------
# PRICE PREDICTION
# -------------------------------
if feature == "Price Prediction":
    st.title("📈 Predict Travel Price")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 100, 30)
        distance = st.number_input("Distance", 0, 5000, 500)
        days = st.number_input("Days", 1, 30, 3)

    with col2:
        base_price = st.number_input("Base Price", 0, 10000, 200)
        company = st.selectbox("Airline", ["Indigo", "Air India", "SpiceJet"])
        gender = st.selectbox("Gender", ["male", "female"])

    if st.button("Predict Price"):
        payload = {
            "age": age,
            "distance": distance,
            "price_y": base_price,
            "time": 5,
            "days": days,
            "place": "Delhi",
            "price_x": 150,
            "company": company,
            "gender": gender,
            "from": "Mumbai",
            "to": "Delhi",
            "flightType": "Economy",
            "agency": "MakeMyTrip"
        }

        res = requests.post(f"{API_URL}/predict_price", json=payload)

        if res.status_code == 200:
            data = res.json()
            if "predicted_total_price" in data:
                st.success(f"💰 Predicted Price: {round(data['predicted_total_price'], 2)}")
            else:
                st.error(data)
        else:
            st.error("API Error")


# -------------------------------
# GENDER PREDICTION
# -------------------------------
elif feature == "Gender Prediction":
    st.title("🧑 Predict Gender")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 100, 30)
        company = st.selectbox("Airline", ["Indigo", "Air India", "SpiceJet"])

    with col2:
        from_loc = st.text_input("From", "Mumbai")
        to_loc = st.text_input("To", "Delhi")

    flight_type = st.selectbox("Flight Type", ["Economy", "Business"])
    agency = st.selectbox("Agency", ["MakeMyTrip", "Goibibo", "Yatra"])

    if st.button("Predict Gender"):

        payload = {
            "age": age,
            "company": company,
            "from_location": from_loc,
            "to_location": to_loc,
            "flightType": flight_type,
            "agency": agency
        }

        res = requests.post(f"{API_URL}/predict_gender", json=payload)

        if res.status_code == 200:
            data = res.json()

            if "predicted_gender" in data:
                st.success(f"👤 Predicted Gender: {data['predicted_gender']}")
            else:
                st.error(data)
        else:
            st.error("API Error")


# -------------------------------
# RECOMMENDATION
# -------------------------------
else:
    st.title("🎯 Travel Recommendations")

    user_id = st.number_input("User ID", 1, 1000, 1)

    if st.button("Get Recommendations"):
        res = requests.get(f"{API_URL}/recommend?user_id={user_id}")

        if res.status_code == 200:
            data = res.json()

            if "recommendations" in data:
                st.success("✨ Recommendations:")
                for i, rec in enumerate(data["recommendations"], 1):
                    st.write(f"{i}. {rec}")
            else:
                st.error(data)
        else:
            st.error("API Error")
