import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
st.set_page_config(page_title="Dashboard", page_icon="🚗", layout="wide")

st.title("🚗 Car Health Dashboard")

# 🔗 رابط البيانات من GitHub (عدل الرابط حسب ريبوك)
url = "https://github.com/Yosranaser/gradution_project/blob/main/predictive_maintenance_final_fixed.csv"

# قراءة البيانات
response = requests.get(url)
data = response.json()

st.subheader("🔧 حالة المكونات الرئيسية")

# 🔥 تقسيم الصفحة إلى 3 أعمدة
col1, col2, col3 = st.columns(3)

col1.metric("ESP32 Temp (°C)", f"{data['esp32_temperature_(°c)']} °C")
col2.metric("Servo Temp (°C)", f"{data['servo_temperature_(°c)']} °C")
col3.metric("Motor Driver Temp (°C)", f"{data['motor_driver_temperature_(°c)']} °C")

col1.metric("STM32 Voltage (V)", f"{data['stm32_voltage_(v)']} V")
col2.metric("Universal Voltage (V)", f"{data['universal_voltage_(v)']} V")
col3.metric("Servo Vibration (g)", f"{data['servo_vibration_(g)']} g")

# 🚨 تنبيه الإشارة المفقودة
if data["ultrasonic_signal_loss"] > 0:
    st.error(f"🚨 Ultrasonic Signal Loss Detected: {data['ultrasonic_signal_loss']}")
else:
    st.success("✅ No Signal Loss Detected in Ultrasonic Sensor")

# 🔥 رسم Gauge لدرجة حرارة ESP32
fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=data['esp32_temperature_(°c)'],
    delta={'reference': 70},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 60], 'color': "lightgreen"},
            {'range': [60, 85], 'color': "orange"},
            {'range': [85, 100], 'color': "red"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 85
        }
    },
    title={'text': "ESP32 Temperature (°C)"}
))

st.plotly_chart(fig, use_container_width=True)

# 📊 جدول بيانات ملخص
st.subheader("📜 Summary Data")

data_table = pd.DataFrame(list(data.items()), columns=["Component", "Value"])
st.dataframe(data_table)



