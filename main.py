import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Food Delivery Time Predictor 🛵")

# --- Load the model and encoders ---

# Load the best Random Forest model
with open('best_random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dictionary of label encoders
with open('all_label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# --- Streamlit UI ---
st.title('🛵 Food Delivery Time Predictor')
st.markdown('### Predict the estimated delivery time for your food orders.')

st.write("--- ") # Separator

# Arrange inputs in columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    distance_km = st.slider('Distance (km)', min_value=0.5, max_value=20.0, value=10.0, step=0.1)
    preparation_time_min = st.slider('Preparation Time (minutes)', min_value=5, max_value=30, value=15, step=1)

with col2:
    courier_experience_yrs = st.slider('Courier Experience (years)', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    weather_options = label_encoders['Weather'].classes_
    weather = st.selectbox('Weather Conditions', weather_options)

with col3:
    traffic_level_options = label_encoders['Traffic_Level'].classes_
    traffic_level = st.selectbox('Traffic Level', traffic_level_options)
    time_of_day_options = label_encoders['Time_of_Day'].classes_
    time_of_day = st.selectbox('Time of Day', time_of_day_options)
    vehicle_type_options = label_encoders['Vehicle_Type'].classes_
    vehicle_type = st.selectbox('Vehicle Type', vehicle_type_options)

st.write("--- ") # Separator

# Prediction button
if st.button('Predict Delivery Time', use_container_width=True, type="primary"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Distance_km': [distance_km],
        'Weather': [weather],
        'Traffic_Level': [traffic_level],
        'Time_of_Day': [time_of_day],
        'Vehicle_Type': [vehicle_type],
        'Preparation_Time_min': [preparation_time_min],
        'Courier_Experience_yrs': [courier_experience_yrs]
    })

    # Apply Label Encoding using the loaded encoders
    for col, encoder in label_encoders.items():
        input_data[col] = encoder.transform(input_data[col])

    # Make prediction
    prediction = model.predict(input_data)

    st.success(f'### Predicted Delivery Time: {prediction[0]:.2f} minutes ⏱️')

st.markdown("""
<br>
<small>Developed with ❤️ using Streamlit and Scikit-learn</small>
""", unsafe_allow_html=True)
