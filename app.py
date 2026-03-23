import streamlit as st
import pandas as pd
import mlflow.pyfunc

st.set_page_config(page_title="Voyage Analytics", layout="centered")

@st.cache_resource
def load_models():
    price_model = mlflow.pyfunc.load_model("models/price_model")
    gender_model = mlflow.pyfunc.load_model("models/gender_model")
    return price_model, gender_model

price_model, gender_model = load_models()

st.title("✈️ Voyage Analytics Dashboard")

from_city = st.text_input("From")
to_city = st.text_input("To")
flight_type = st.selectbox("Flight Type", ["Economy", "Business"])

price_x = st.number_input("Base Price", value=1000.0)
distance = st.number_input("Distance", value=500.0)

travel_month = st.slider("Travel Month", 1, 12, 6)
travel_day = st.slider("Travel Day", 1, 31, 15)

return_month = st.slider("Return Month", 1, 12, 6)
return_day = st.slider("Return Day", 1, 31, 20)

if st.button("Predict"):
    input_data = {
        "from": from_city,
        "to": to_city,
        "flightType": flight_type,
        "price_x": price_x,
        "distance": distance,
        "travel_month": travel_month,
        "travel_day": travel_day,
        "return_month": return_month,
        "return_day": return_day
    }

    df = pd.DataFrame([input_data])

    price_pred = price_model.predict(df)[0]
    gender_pred = gender_model.predict(df)[0]

    st.success(f"💰 Predicted Travel Price: {price_pred:.2f}")
    st.success(f"🧑 Predicted Gender: {gender_pred}")
