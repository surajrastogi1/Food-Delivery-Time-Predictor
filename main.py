import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Food Delivery Time Predictor 🛵")

# --- Load the model and encoders ---
# Using absolute paths to prevent "File Not Found" errors on deployment
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, 'best_random_forest_model.pkl')
encoder_path = os.path.join(BASE_DIR, 'all_label_encoders.pkl')

@st.cache_resource
def load_assets():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

try:
    model, label_encoders = load_assets()
except FileNotFoundError:
    st.error("Model files not found. Please ensure the .pkl files are in the repository.")
    st.stop()

# --- Streamlit UI ---
st.title('🛵 Food Delivery Time Predictor')
st.markdown('### Predict the estimated delivery time for your food orders.')
st.write("---")

# Arrange inputs in columns
col1, col2, col3 = st.columns(3)

with col1:
    distance_km = st.slider('Distance (km)', 0.5, 20.0, 10.0, 0.1)
    preparation_time_min = st.slider('Preparation Time (minutes)', 5, 30, 15)

with col2:
    courier_experience_yrs = st.slider('Courier Experience (years)', 0.0, 10.0, 5.0, 0.1)
    weather = st.selectbox('Weather Conditions', label_encoders['Weather'].classes_)

with col3:
    traffic_level = st.selectbox('Traffic Level', label_encoders['Traffic_Level'].classes_)
    time_of_day = st.selectbox('Time of Day', label_encoders['Time_of_Day'].classes_)
    vehicle_type = st.selectbox('Vehicle Type', label_encoders['Vehicle_Type'].classes_)

st.write("---")

# Prediction button
if st.button('Predict Delivery Time', use_container_width=True, type="primary"):
    # 1. Create Input DataFrame
    input_data = pd.DataFrame({
        'Distance_km': [distance_km],
        'Weather': [weather],
        'Traffic_Level': [traffic_level],
        'Time_of_Day': [time_of_day],
        'Vehicle_Type': [vehicle_type],
        'Preparation_Time_min': [preparation_time_min],
        'Courier_Experience_yrs': [courier_experience_yrs]
    })

    # 2. Apply Label Encoding only to the categorical columns
    categorical_cols = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
    
    for col in categorical_cols:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # 3. Make prediction
    # Ensure the column order matches exactly how the model was trained
    prediction = model.predict(input_data)

    # 4. Display Result
    st.success(f'### Predicted Delivery Time: {prediction[0]:.2f} minutes ⏱️')
    
st.markdown("""
<br><center><small>Developed with ❤️ using Streamlit and Scikit-learn</small></center>
""", unsafe_allow_html=True)
