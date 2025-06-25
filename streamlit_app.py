import streamlit as st
import os
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
import cv2
import io
import pickle
import json
st.set_page_config(layout="wide")
col1, col2 = st.columns([2, 1])  
with col2:
    st.markdown("""
    <div style="background-color:#f2f2f2; padding:25px; border-radius:15px; text-align:right; direction:rtl;">
        <h3 style="color:#1f77b4;">👋 إزّيك؟</h3>
        <p style="font-size:18px;">الموقع ده معمول علشان يساعدك تعرف حالة عربيتك بسهولة.</p>

        <ul style="font-size:16px;">
            <li>لو العربية فيها مشكلة أو محتاجة صيانة، هنقولك فورًا.</li>
            <li>تقدر كمان تشوف <b>نسبة البنزين</b>، <b>السرعة</b>، و<b>درجة حرارة الأجزاء المهمة</b> وانت سايق أو قبل ما تتحرك.</li>
        </ul>

        <p style="font-size:17px;">
            كل اللي عليك:
            <br>✅ تقول <b>"صيانة"</b> لو حابب تتطمن على حالة العربية.
            <br>✅ أو تقول <b>"عرض البيانات"</b> لو حابب تشوف كل حاجة شغالة إزاي دلوقتي.
        </p>

        <p style="font-size:17px; color:#d62728;"><b>📷 ولكن يجب التحقق من الشخصية أولاً. التقط صورة لك.</b></p>
    </div>
    """, unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your Firebase serviceAccountKey.json", type="json")

if uploaded_file is not None:
    with open("temp_firebase_key.json", "wb") as f:
        f.write(uploaded_file.getbuffer())

cred = credentials.Certificate("temp_firebase_key.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
            'databaseURL':'https://console.firebase.google.com/project/car-data-9c9fc/database/car-data-9c9fc-default-rtdb/data'
        })

    st.success("✅ Firebase Connected Successfully!")

   
    esp32_temperature = db.reference('esp32_temperature_(°c)').get()
    stm32_voltage = db.reference('stm32_voltage_(v)').get()
    stm32_temperature = db.reference('stm32_temperature_(°c)').get()
    servo_temperature = db.reference('servo_temperature_(°c)').get()
    ultrasonic_voltage = db.reference('ultrasonic_voltage_(v)').get()
    motor_driver_temperature = db.reference('motor_driver_temperature_(°c)').get()
    servo_voltage = db.reference('servo_voltage_(v)').get()
    servo_vibration = db.reference('servo_vibration_(g)').get()
    universal_voltage = db.reference('universal_voltage_(v)').get()
    motor_driver_voltage = db.reference('motor_driver_voltage_(v)').get()
    servo_motor_voltage = db.reference('servo_motor_voltage_(v)').get()
    universal_motor_voltage = db.reference('universal_motor_voltage_(v)').get()
    ultrasonic_signal_loss = db.reference('ultrasonic_signal_loss').get()
    universal_current = db.reference('universal_current_(a)').get()
    universal_motor_current = db.reference('universal_motor_current_(a)').get()
    stm32_current = db.reference('stm32_current_(a)').get()
    ultrasonic_temperature = db.reference('ultrasonic_temperature_').get()
    motor_driver_current = db.reference('motor_driver_current_(a)').get()
    servo_motor_current = db.reference('servo_motor_current_(a)').get()
    universal_noise = db.reference('universal_noise_(db)').get()
    servo_current = db.reference('servo_current_(a)').get()
    esp32_voltage = db.reference('esp32_voltage_(v)').get()
    esp32_current = db.reference('esp32_current_(a)').get()
    stm_temperature = db.reference('stm_temperature_(°c)').get()
    universal_temperature = db.reference('universal_temperature_(°c)').get()
    try:
        data = {
            "esp32_temperature_(°c)": db.reference('esp32_temperature_(°c)').get(),
            "stm32_voltage_(v)": db.reference('stm32_voltage_(v)').get(),
            "stm32_temperature_(°c)": db.reference('stm32_temperature_(°c)').get(),
            "servo_temperature_(°c)": db.reference('servo_temperature_(°c)').get(),
            "ultrasonic_voltage_(v)": db.reference('ultrasonic_voltage_(v)').get(),
            "motor_driver_temperature_(°c)": db.reference('motor_driver_temperature_(°c)').get(),
            "servo_voltage_(v)": db.reference('servo_voltage_(v)').get(),
            "servo_vibration_(g)": db.reference('servo_vibration_(g)').get(),
            "universal_voltage_(v)": db.reference('universal_voltage_(v)').get(),
            "motor_driver_voltage_(v)": db.reference('motor_driver_voltage_(v)').get(),
            "servo_motor_voltage_(v)": db.reference('servo_motor_voltage_(v)').get(),
            "universal_motor_voltage_(v)": db.reference('universal_motor_voltage_(v)').get(),
            "ultrasonic_signal_loss": db.reference('ultrasonic_signal_loss').get(),
            "universal_current_(a)": db.reference('universal_current_(a)').get(),
            "universal_motor_current_(a)": db.reference('universal_motor_current_(a)').get(),
            "stm32_current_(a)": db.reference('stm32_current_(a)').get(),
            "ultrasonic_temperature_": db.reference('ultrasonic_temperature_').get(),
            "motor_driver_current_(a)": db.reference('motor_driver_current_(a)').get(),
            "servo_motor_current_(a)": db.reference('servo_motor_current_(a)').get(),
            "universal_noise_(db)": db.reference('universal_noise_(db)').get(),
            "servo_current_(a)": db.reference('servo_current_(a)').get(),
            "esp32_voltage_(v)": db.reference('esp32_voltage_(v)').get(),
            "esp32_current_(a)": db.reference('esp32_current_(a)').get(),
            "stm_temperature_(°c)": db.reference('stm_temperature_(°c)').get(),
            "universal_temperature_(°c)": db.reference('universal_temperature_(°c)').get()
        }
    
    df = pd.DataFrame([data])
    st.dataframe(df)
    with open('model (2).pkl', 'rb') as file:
       model = pickle.load(file)
            
          
    if st.button("🔍 Predict Status"):
        prediction = model.predict(df)[0]
        st.subheader(f"⚙️ Prediction Result: **{prediction}**")
                
