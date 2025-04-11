import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from meteostat import Point, Daily
import numpy as np

# Title
st.title("Weather Forecast Dashboard")

# Location input
city = st.text_input("Enter City Name", "New Delhi")
lat = st.number_input("Latitude", value=28.6139)
lon = st.number_input("Longitude", value=77.2090)

# Load model and scaler
model = load_model("lstm_temperature_forecast_model.h5")
scaler = joblib.load("scaler_temperature.save")

# Date range for prediction
days = st.slider("Select number of past days", min_value=7, max_value=60, value=30)
today = datetime.now().date()
start = today - timedelta(days=days)
end = today

# Fetch data
data_load_state = st.text("Fetching weather data...")
location = Point(lat, lon)
data = Daily(location, start, end).fetch().reset_index()
data.rename(columns={
    "tavg": "temperature",
    "wspd": "wind_speed",
    "pres": "pressure",
    "prcp": "precipitation"
}, inplace=True)
data = data[["time", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]]
data.rename(columns={"time": "date"}, inplace=True)
data.dropna(inplace=True)
data_load_state.text("Weather data loaded.")

# Show data
display_data = st.checkbox("Show raw weather data")
if display_data:
    st.dataframe(data)

# Preprocessing
X = scaler.transform(data.drop(columns=["date"]))
predictions = model.predict(X).flatten()
data["predicted_temperature"] = predictions

# Visualize
st.subheader("Actual vs Predicted Temperature")
st.line_chart(data.set_index("date")[[
    "temperature", "predicted_temperature"
]])

# Save as CSV
download = st.download_button(
    label="Download Predictions as CSV",
    data=data.to_csv(index=False).encode('utf-8'),
    file_name="predicted_weather.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Meteostat by SHUBHAM and TEAM")