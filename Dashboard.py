import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Dashboard", page_icon="🚗", layout="wide")

st.title("🚗 Car Health Dashboard")

# قراءة البيانات من GitHub
url = "https://raw.githubusercontent.com/username/repo/main/data.json"
response = requests.get(url)
data = response.json()

st.subheader("🔧 بيانات السيارة")
st.json(data)

esp32_temp = data['esp32_temperature_(°c)']
st.metric("ESP32 Temperature", f"{esp32_temp}°C")

# باقي التصميم والرسومات هنا...

