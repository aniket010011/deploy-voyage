import streamlit as st
import requests

# ---------------- CONFIG ---------------- #
API_URL = "http://a6720fa00f3da44019c760cf5cdd2607-798674393.eu-north-1.elb.amazonaws.com"

st.set_page_config(page_title="Voyage Analytics", page_icon="✈️", layout="wide")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("✈️ Voyage Analytics")
page = st.sidebar.radio(
    "Select Feature",
    ["📈 Price Prediction", "🧑 Gender Prediction", "🎯 Recommendation"]
)

st.title("✈️ Voyage Analytics Platform")

# ---------------- PRICE PREDICTION ---------------- #
if page == "📈 Price Prediction":

    st.header("📈 Predict Travel Price")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", value=30)
        distance = st.number_input("Distance", value=500)
        days = st.number_input("Days", value=3)

    with col2:
        price = st.number_input("Base Price", value=200)
        airline = st.selectbox("Airline", ["Indigo", "Air India", "SpiceJet"])
        gender = st.selectbox("Gender", ["male", "female"])

    if st.button("Predict Price"):

        payload = {
            "age": age,
            "distance": distance,
            "price_y": price,
            "time": 5,
            "days": days,
            "place": "Delhi",
            "price_x": 150,
            "company": airline,
            "gender": gender,
            "from_location": "Mumbai",   # ✅ FIXED
            "to_location": "Delhi",      # ✅ FIXED
            "flightType": "Economy",
            "agency": "MakeMyTrip"
        }

        with st.spinner("Predicting..."):
            res = requests.post(f"{API_URL}/predict_price", json=payload)

        if res.status_code == 200:
            result = res.json()["predicted_total_price"]
            st.success(f"💰 Predicted Price: {result:.2f}")
        else:
            st.error(res.text)

# ---------------- GENDER ---------------- #
elif page == "🧑 Gender Prediction":

    st.header("🧑 Predict Gender")

    age = st.number_input("Age", value=30)
    company = st.selectbox("Airline", ["Indigo", "Air India", "SpiceJet"])

    if st.button("Predict Gender"):

        payload = {
            "age": age,
            "company": company
        }

        with st.spinner("Predicting..."):
            res = requests.post(f"{API_URL}/predict_gender", json=payload)

        if res.status_code == 200:
            response = res.json()

            if "gender" in response:
                st.success(f"👤 Predicted Gender: {response['gender']}")
            else:
                st.error(response)

        else:
            st.error(res.text)

# ---------------- RECOMMENDATION ---------------- #
elif page == "🎯 Recommendation":

    st.header("🎯 Travel Recommendations")

    user_id = st.number_input("User ID", value=1)

    if st.button("Get Recommendations"):

        with st.spinner("Fetching recommendations..."):
            res = requests.get(f"{API_URL}/recommend", params={"user_id": user_id})

        if res.status_code == 200:
            response = res.json()

            if "recommendations" in response:
                recs = response["recommendations"]

                if len(recs) == 0:
                    st.warning("No recommendations found")
                else:
                    st.success("✨ Recommendations:")
                    for i, r in enumerate(recs, 1):
                        st.write(f"{i}. {r}")
            else:
                st.error(response)

        else:
            st.error(res.text)
