import streamlit as st
import requests

# The URL of your AWS Load Balancer (ensure it's the standard HTTP link)
BASE_URL = "http://a6720fa00f3da44019c760cf5cdd2607-798674393.eu-north-1.elb.amazonaws.com"

st.title("Voyage Analytics Dashboard (Cloud API)")

tab1, tab2, tab3 = st.tabs(["Price Prediction", "Gender Classification", "Recommendations"])

with tab1:
    st.header("Predict Travel Price")
    with st.form("price_form"):
        # Input fields matching your Swagger data structure
        age = st.number_input("Age", value=30)
        distance = st.number_input("Distance", value=500)
        price_y = st.number_input("Price Y", value=200)
        time = st.number_input("Time", value=5)
        days = st.number_input("Days", value=3)
        place = st.text_input("Place", value="Delhi")
        price_x = st.number_input("Price X", value=150)
        company = st.selectbox("Airline", ["Indigo", "AirIndia", "Vistara"])
        gender = st.selectbox("Gender", ["male", "female"])
        from_loc = st.text_input("From", value="Mumbai")
        to_loc = st.text_input("To", value="Delhi")
        flightType = st.selectbox("Class", ["Economy", "Business"])
        agency = st.text_input("Agency", value="MakeMyTrip")
        
        submit_price = st.form_submit_button("Predict")
        
        if submit_price:
            payload = {
                "age": age, "distance": distance, "price_y": price_y, "time": time,
                "days": days, "place": place, "price_x": price_x, "company": company,
                "gender": gender, "from": from_loc, "to": to_loc, 
                "flightType": flightType, "agency": agency
            }
            try:
                response = requests.post(f"{BASE_URL}/predict_price", json=payload)
                if response.status_code == 200:
                    res_data = response.json()
                    st.success(f"Predicted Price: ${res_data.get('predicted_price', 'N/A')}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

with tab2:
    st.header("Classify Gender")
    # Add similar form for /predict_gender if needed...
    st.info("Uses the same request logic as above hitting /predict_gender")

with tab3:
    st.header("Recommendations")
    user_id = st.number_input("User ID", min_value=1, value=1)
    if st.button("Get Recommendations"):
        try:
            # Note: Ensure your FastAPI has a /recommend endpoint
            response = requests.get(f"{BASE_URL}/recommend/{user_id}")
            if response.status_code == 200:
                st.write(response.json())
            else:
                st.warning("Recommendation endpoint not found on AWS.")
        except Exception as e:
            st.error(f"Failed: {e}")
