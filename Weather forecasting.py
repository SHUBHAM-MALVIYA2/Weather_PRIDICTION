#!/usr/bin/env python
# coding: utf-8

# ### 1. Data Collection (Meteostat API)
# We’ll use the meteostat Python package to pull historical weather data.

# In[1]:


from meteostat import Point, Daily
import pandas as pd
from datetime import datetime

# Define location (Example: Delhi, India)
location = Point(28.6139, 77.2090)  # Latitude & Longitude

# Time range
start = datetime(2022, 1, 1)
end = datetime(2023, 12, 31)

# Get daily data
data = Daily(location, start, end)
data = data.fetch()

# Save to CSV
data.to_csv("weather_data.csv")

print("Dataset downloaded and saved as weather_data.csv")


#  ### Step-by-Step Data Cleaning (Python)
# ✅ 1. Load the Dataset Properly

# In[2]:


import pandas as pd

# Load the CSV file
df = pd.read_csv("weather_data.csv")

# Convert 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])


# ✅ 2. Check the Dataset Info

# In[3]:


print(df.info())         # Data types and non-null counts
print(df.head())         # Peek at the data
print(df.isnull().sum()) # Missing values per column


# ### Handle Missing Values
# You can either drop or fill them depending on how much data is missing.
# 
# 
# ✅ Best for time series: Use ffill then bfill to avoid losing data unless missing values are excessive.

# In[4]:


df_cleaned = df.fillna(method='ffill')  # Fill using previous row
df_cleaned = df_cleaned.fillna(method='bfill')  # Fill from next row if still missing


# ✅ 4. Ensure Data Types Are Numeric Where Needed

# In[5]:


numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
print("Numeric columns:", numeric_columns)

# Check for any stray non-numeric data
print(df_cleaned.dtypes)


# #### Reset Index and Save Cleaned Data

# In[7]:


df_cleaned = df_cleaned.reset_index(drop=True)
df_cleaned.to_csv("cleaned_weather_data.csv", index=False)
print("Cleaned dataset saved as cleaned_weather_data.csv")


# ## Step-by-Step: Outlier Detection + Visualization
# Step 1: Load and Preprocess the Data

# In[8]:


import pandas as pd

df = pd.read_csv("weather_data.csv")
df['time'] = pd.to_datetime(df['time'])
df = df.fillna(method='ffill').fillna(method='bfill')  # Handle missing values


# 🚨 Step 4: Detecting Outliers Using IQR

# In[11]:


def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# Apply to selected columns
cols_to_check = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd']
df_no_outliers = remove_outliers_iqr(df, cols_to_check)


# Step 5: Boxplots to Visualize Outliers

# In[12]:


plt.figure(figsize=(14, 6))
sns.boxplot(data=df[cols_to_check])
plt.title("Boxplot Before Removing Outliers")
plt.show()

# After removing
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_no_outliers[cols_to_check])
plt.title("Boxplot After Removing Outliers")
plt.show()


# Interactive Plot (Plotly)
# 

# In[14]:


pip install plotly


# In[15]:


import plotly.express as px

fig = px.line(df, x='time', y='tavg', title='Average Temperature (Interactive)')
fig.show()


# 💾 Save your clean data:

# In[17]:


df_no_outliers.to_csv("cleaned_no_outliers.csv", index=False)


# ### 3. Model: LSTM for Weather Forecasting

# In[18]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Assume df is your DataFrame with weather features
def create_dataset(series, time_steps=7):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i:i + time_steps])
        y.append(series[i + time_steps])
    return np.array(X), np.array(y)

# Example: Predict avg temp using past 7 days
target = df['tavg'].values.reshape(-1, 1)
scaler = MinMaxScaler()
target_scaled = scaler.fit_transform(target)

X, y = create_dataset(target_scaled)
X = X.reshape((X.shape[0], X.shape[1], 1))  # [samples, time_steps, features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# LSTM Model
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)


#  ### Step 4: Forecasting with Deep Learning (LSTM)
# We'll predict next day's temperature using past 7 days’ data.

# In[19]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Use cleaned data without outliers
df = df_no_outliers.copy()

# Use average temperature ('tavg') as target
target_col = 'tavg'

# Scale the data
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df[[target_col]])

# Create sequences
def create_sequences(data, time_steps=7):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_values)

# Reshape X for LSTM [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))


# In[20]:


# split the model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)


# In[21]:


# Predict on test set
y_pred = model.predict(X_test)

# Inverse scale
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Compare actual vs predicted
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="Actual")
plt.plot(y_pred_rescaled, label="Predicted")
plt.legend()
plt.title("Temperature Prediction: Actual vs Predicted")
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.grid()
plt.show()


# ###  Save the Trained LSTM Model
# We'll use Keras' built-in saving functionality.

# In[22]:


# Save the model in HDF5 format
model.save("lstm_temperature_forecast_model.h5")
print("✅ Model saved as lstm_temperature_forecast_model.h5")


# ### 🔁 Load the Model Later
# When you need to load and use it again:

# In[23]:


from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("lstm_temperature_forecast_model.h5")

# Predict as usual
y_pred = model.predict(X_test)


# #### Save the Scaler (for inverse transform later)
# Use joblib or pickle:

# In[24]:


import joblib

# Save the scaler
joblib.dump(scaler, "scaler_temperature.save")
print("✅ Scaler saved as scaler_temperature.save")


# In[25]:


scaler = joblib.load("scaler_temperature.save")


# #### Pipeline

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os

# ============ Step 1: Load and Clean Dataset ============
df = pd.read_csv("weather_data.csv")
df['time'] = pd.to_datetime(df['time'])
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# ============ Step 2: Outlier Removal ============
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

cols_to_check = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd']
df = remove_outliers_iqr(df, cols_to_check)
df.reset_index(drop=True, inplace=True)

# ============ Step 3: Prepare Data for LSTM ============
target_col = 'tavg'
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[[target_col]])

def create_sequences(data, time_steps=7):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ============ Step 4: Build & Train LSTM Model ============
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1, verbose=1)

# ============ Step 5: Save Model & Scaler ============
model.save("lstm_temperature_forecast_model.h5")
joblib.dump(scaler, "scaler_temperature.save")
print("✅ Model and scaler saved.")

# ============ Step 6: Forecast and Save Results ============
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

forecast_df = pd.DataFrame({
    'date': df['time'].iloc[-len(y_test):].reset_index(drop=True),
    'actual_temperature': y_test_inv.flatten(),
    'predicted_temperature': y_pred_inv.flatten()
})

forecast_df.to_csv("temperature_forecast.csv", index=False)
print("📈 Forecast saved to temperature_forecast.csv")

# ============ Step 7: Plot Result ============
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Actual")
plt.plot(y_pred_inv, label="Predicted")
plt.legend()
plt.title("Temperature Forecast: Actual vs Predicted")
plt.xlabel("Time Step")
plt.ylabel("Temperature (°C)")
plt.grid()
plt.tight_layout()
plt.savefig("forecast_plot.png")
plt.show()


# In[28]:


#### ✅ Files Generated
###
#lstm_temperature_forecast_model.h5: Your trained model

####scaler_temperature.save: Scaler for future predictions

###temperature_forecast.csv: Output for Supaboard/Supabase

####forecast_plot.png: Visualization to show on dashboard



# #### Integration

# In[ ]:





# In[50]:


# piplining
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import requests
import os

# === CONFIG ===
SUPABASE_URL = "https://fwbnqmoguwwpoykjnpab.supabase.co"  # <-- replace this
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ3Ym5xbW9ndXd3cG95a2pucGFiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQzODMzOTQsImV4cCI6MjA1OTk1OTM5NH0.QBAE3hyFieWhKdF575VBfLhRTIWMVml--b61S7oVWLw"  # <-- replace this
TABLE_NAME = "forecast"
CSV_PATH = "weather_data.csv"

# === Step 1: Load + Clean Data ===
df = pd.read_csv(CSV_PATH)
df['time'] = pd.to_datetime(df['time'])
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# === Step 2: Remove Outliers ===
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

df = remove_outliers_iqr(df, ['tavg', 'tmin', 'tmax', 'prcp', 'wspd'])
df.reset_index(drop=True, inplace=True)

# === Step 3: Prepare for LSTM ===
target_col = 'tavg'
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[[target_col]])

def create_sequences(data, time_steps=7):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)
X = X.reshape((X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Step 4: Train LSTM Model ===
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1, verbose=1)

# === Step 5: Save Artifacts ===
model.save("lstm_temperature_forecast_model.h5")
joblib.dump(scaler, "scaler_temperature.save")

# === Step 6: Forecast + Save CSV ===
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

forecast_df = pd.DataFrame({
    'date': df['time'].iloc[-len(y_test):].reset_index(drop=True),
    'actual_temperature': y_test_inv.flatten(),
    'predicted_temperature': y_pred_inv.flatten()
})
forecast_df.to_csv("temperature_forecast.csv", index=False)

# === Step 7: Upload to Supabase ===
forecast_df['date'] = forecast_df['date'].astype(str)  # Convert Timestamp to ISO string
records = forecast_df.to_dict(orient='records')

headers = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}
insert_url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"

print("\n🌐 Uploading data to Supabase...")
for record in records:
    response = requests.post(insert_url, json=record, headers=headers)
    if response.status_code not in [200, 201]:
        print(f"❌ Failed to insert: {record}")
        print(response.text)
    else:
        print(f"✅ Uploaded: {record['date']}")

# === Step 8: Plot the Forecast ===
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label="Actual")
plt.plot(y_pred_inv, label="Predicted")
plt.title("Temperature Forecast - Actual vs Predicted")
plt.xlabel("Time Steps")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("forecast_plot.png")
plt.show()


# In[ ]:




