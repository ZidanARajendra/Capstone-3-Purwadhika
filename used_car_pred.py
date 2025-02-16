# Import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Judul Utama
st.title('Saudi Arabia Used Car Price Predictor')
st.text('This web can be used to predict the price of a used car')

# Menambahkan sidebar
st.sidebar.header("Please input car features")

def create_user_input():
    
    # Numerical Features
    year = st.sidebar.slider('Year', min_value=1963, max_value=2021, value=2015)
    engine_size = st.sidebar.slider('Engine Size', min_value=1.0, max_value=9.0, value=2.0, step=0.1)
    mileage = st.sidebar.number_input('Mileage', min_value=100, max_value=1_000_000, value=50_000)

    # Categorical Features
    car_type = st.sidebar.selectbox('Car Type', [
        'Optima', 'CX3', 'Sonata', 'Avalon', 'Land Cruiser', 'FJ', 'Tucson', 'Sunny', 'Azera', 'Pathfinder', 'Accent', 'Corolla', 
        'Altima', 'Senta fe', 'Land Cruiser Pickup', 'VTC', 'Patrol', 'Camry', 'Previa', 'Datsun', 'Hilux', '6', 'Innova', 'Navara', 
        'Carnival', 'Elantra', 'Cerato', 'Furniture', 'Murano', 'Land Cruiser 70', '3', 'Hiace', 'CX9', 'Yaris', 'Sylvian Bus', 'Opirus', 
        'Creta', 'Sedona', 'Cores', 'Cadenza', 'Rio', 'Maxima', 'X-Trail', 'Prado', 'H1', 'Rav4', 'Genesis', 'CX5', 'Mohave', 'Rush', 'Sentra', 
        'Veloster', 'Ciocca', 'Kona', 'Sorento', 'Carenz', 'Avanza', 'Coupe S', 'Juke', 'Sportage', 'X-Terra', 'Picanto', 'KICKS', 'Other', 
        'Aurion', 'Bus Urvan', 'Seltos', 'Prius', 'Cressida', 'Armada', 'Pegas', 'Coaster', 'Z370', 'Bus County', 'Stinger', 'K5', 'Carens', 
        'Tuscani', '4Runner', '2', 'i40', 'Soul', 'Avante', 'Z350', 'CX7'
    ])
    
    make = st.sidebar.selectbox('Make', [
        'Toyota', 'Hyundai', 'Nissan', 'Mazda', 'Kia'
    ])
    
    region = st.sidebar.selectbox('Region', [
        'Riyadh', 'Jeddah', 'Dammam', 'Makkah', 'Medina', 'Tabouk', 'Yanbu', 'Hail', 'Abha'
    ])
    
    gear_type = st.sidebar.radio('Gear Type', ['Automatic', 'Manual'])
    
    origin = st.sidebar.radio('Origin', ['Saudi', 'Gulf Arabic', 'Other', 'Unknown'])
    
    options = st.sidebar.radio('Options', ['Full', 'Semi Full', 'Standard'])

    # Creating a dictionary with user input
    user_data = {
        'Year': year,
        'Engine_Size': engine_size,
        'Mileage': mileage,
        'Type': car_type,
        'Make': make,
        'Region': region,
        'Gear_Type': gear_type,
        'Origin': origin,
        'Options': options
    }
    
    # Convert the dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df

# Get car data from user
data_car = create_user_input()

# Membuat 2 kontainer
col1, col2 = st.columns(2)

# Kiri
with col1:
    st.subheader("Car's Features")
    st.write(data_car.transpose())

# Load model
with open('final_model.sav', 'rb') as f:
    model_loaded = pickle.load(f)
    
# Predict price
predicted_price = model_loaded.predict(data_car)
    
# Menampilkan hasil prediksi

# Bagian kanan (col2)
with col2:
    st.subheader('Predicted Price')
    st.write(f"The estimated price of this car is: **{predicted_price[0]:,.2f} SAR**")
