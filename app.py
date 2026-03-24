import streamlit as st
import requests

# The URL of your AWS Load Balancer
BASE_URL = "http://a6720fa00f3da44019c760cf5cdd2607-798674393.eu-north-1.elb.amazonaws.com"

st.set_page_config(page_title="Voyage Analytics Dashboard", layout="wide")
st.title("Voyage Analytics Dashboard (Cloud API)")

tab1, tab2, tab3 = st.tabs(["Price Prediction", "Gender Classification", "Recommendations"])

# --- TAB 1: PRICE PREDICTION ---
with tab1:
    st.header("Predict Travel Price")
    with st.form("price_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", value=30, key="price_age")
            distance = st.number_input("Distance", value=500, key="price_dist")
            price_y = st.number_input("Price Y", value=200, key="price_py")
            time = st.number_input("Time", value=5, key="price_time")
            days = st.number_input("Days", value=3, key="price_days")
            place = st.text_input("Place", value="Delhi", key="price_place")
        with col2:
            price_x = st.number_input("Price X", value=150, key="price_px")
            company = st.selectbox("Airline", ["Indigo", "AirIndia", "Vistara"], key="price_airline")
            from_loc = st.text_input("From", value="Mumbai", key="price_from")
            to_loc = st.text_input("To", value="Delhi", key="price_to")
            flightType = st.selectbox("Class", ["Economy", "Business"], key="price_class")
            agency = st.text_input("Agency", value="MakeMyTrip", key="price_agency")
        
        submit_price = st.form_submit_button("Predict Price")
        
        if submit_price:
            payload = {
                "age": age, "distance": distance, "price_y": price_y, "time": time,
                "days": days, "place": place, "price_x": price_x, "company": company,
                "from": from_loc, "to": to_loc, "flightType": flightType, "agency": agency
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

# --- TAB 2: GENDER CLASSIFICATION ---
with tab2:
    st.header("Classify Gender")
    st.write("Predict the traveler's gender based on flight and travel details.")
    
    with st.form("gender_form"):
        col1, col2 = st.columns(2)
        with col1:
            g_age = st.number_input("Age", value=30, key="gen_age")
            g_dist = st.number_input("Distance", value=500, key="gen_dist")
            g_price_y = st.number_input("Price Y", value=200, key="gen_py")
            g_time = st.number_input("Time", value=5, key="gen_time")
            g_days = st.number_input("Days", value=3, key="gen_days")
        with col2:
            g_place = st.text_input("Place", value="Delhi", key="gen_place")
            g_price_x = st.number_input("Price X", value=150, key="gen_px")
            g_company = st.selectbox("Airline", ["Indigo", "AirIndia", "Vistara"], key="gen_airline")
            g_from = st.text_input("From", value="Mumbai", key="gen_from")
            g_to = st.text_input("To", value="Delhi", key="gen_to")
            g_type = st.selectbox("Class", ["Economy", "Business"], key="gen_class")
            g_agency = st.text_input("Agency", value="MakeMyTrip", key="gen_agency")
            
        submit_gender = st.form_submit_button("Classify Gender")
        
        if submit_gender:
            # Note: The backend build_full_features handles the missing 'total' and date fields
            payload = {
                "age": g_age, "distance": g_dist, "price_y": g_price_y, "time": g_time,
                "days": g_days, "place": g_place, "price_x": g_price_x, "company": g_company,
                "from": g_from, "to": g_to, "flightType": g_type, "agency": g_agency
            }
            try:
                response = requests.post(f"{BASE_URL}/predict_gender", json=payload)
                if response.status_code == 200:
                    res_data = response.json()
                    gender_result = res_data.get('predicted_gender', 'N/A')
                    st.success(f"Predicted Gender: **{gender_result.upper()}**")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

# --- TAB 3: RECOMMENDATIONS ---
with tab3:
    st.header("Recommendations")
    user_id = st.number_input("User ID", min_value=1, value=1)
    if st.button("Get Recommendations"):
        try:
            response = requests.get(f"{BASE_URL}/recommend/{user_id}")
            if response.status_code == 200:
                st.json(response.json())
            else:
                st.warning("Recommendation endpoint error.")
        except Exception as e:
            st.error(f"Failed: {e}")
