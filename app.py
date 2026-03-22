import streamlit as st
import requests

API_URL = "http://a6720fa00f3da44019c760cf5cdd2607-798674393.eu-north-1.elb.amazonaws.com"

st.title("✈️ Voyage Analytics Demo")

# Inputs
age = st.number_input("Age", value=30)
distance = st.number_input("Distance", value=500)
price = st.number_input("Base Price", value=200)
days = st.number_input("Days", value=3)

if st.button("Predict Price"):
    payload = {
        "age": age,
        "distance": distance,
        "price_y": price,
        "time": 5,
        "days": days,
        "place": "Delhi",
        "price_x": 150,
        "company": "Indigo",
        "gender": "male",
        "from": "Mumbai",
        "to": "Delhi",
        "flightType": "Economy",
        "agency": "MakeMyTrip"
    }

    response = requests.post(f"{API_URL}/predict_price", json=payload)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Price: {result['predicted_total_price']}")
    else:
        st.error("API Error")