import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(layout="wide")

# 🎨 الواجهة
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

# 🚗 تحميل البيانات من Google Sheet
sheet_id = "10GFBlxh8nNU-yIe7_UH0O6UDqW4Uv_fc0zNR_xC_O00"
sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
df = pd.read_csv(sheet_url)
df = df.T
df.columns = df.iloc[0]  # أول صف يتحول إلى أسماء أعمدة
df = df.drop(df.index[0])  # حذف أول صف من البيانات لأنه بقى أسماء الأعمدة
df = df.reset_index(drop=True)
# ✅ تحويل القيم لأرقام لو محتاجه
df = df.apply(pd.to_numeric, errors='ignore')

# ✅ عرض الجدول في Streamlit
st.title("📊 بيانات السيارة بعد تحويل الأعمدة إلى صفوف")
st.dataframe(df)

# ✅ تحميل الموديل
with open('model (2).pkl', 'rb') as f:
    model = pickle.load(f)

feature_names = [
    'esp32_temperature_(°c)', 'stm32_voltage_(v)', 'stm32_temperature_(°c)',
    'servo_temperature_(°c)', 'ultrasonic_voltage_(v)', 'motor_driver_temperature_(°c)',
    'servo_voltage_(v)', 'servo_vibration_(g)', 'universal_voltage_(v)',
    'motor_driver_voltage_(v)', 'servo_motor_voltage_(v)', 'universal_motor_voltage_(v)',
    'ultrasonic_signal_loss', 'universal_current_(a)', 'universal_motor_current_(a)',
    'stm32_current_(a)', 'ultrasonic_temperature_', 'motor_driver_current_(a)',
    'servo_motor_current_(a)', 'universal_noise_(db)', 'servo_current_(a)',
    'esp32_voltage_(v)', 'esp32_current_(a)', 'stm_temperature_(°c)',
    'universal_temperature_(°c)', 'speed', 'fuel','timestamp'
]

# ✅ ترتيب الأعمدة بشكل صحيح
try:
    selected_df = selected_df[model.feature_names_in_]
except Exception as e:
    st.error(f"❌ مشكلة في الأعمدة: {e}")

# ✅ تأكد من النوع
try:
    selected_df = selected_df.astype(float)
except Exception as e:
    st.error(f"❌ خطأ في تحويل البيانات لأرقام: {e}")

# ✅ لو عايزة صف واحد
selected_row = selected_df.iloc[[0]]

# ✅ التنبؤ
if st.button("🔍 Predict"):
    prediction = model.predict(selected_row)[0]

    fault_mapping = {
        0: "No Fault",
        1: "Overcurrent",
        2: "Undervoltage",
        3: "Overtemperature",
        4: "Ultrasonic Failure",
        5: "Motor Driver Fault",
        6: "ESP32 Overload"
    }

    fault_name = fault_mapping.get(prediction, "Unknown Fault")
    st.subheader(f"⚙️ Prediction Result: **{fault_name}**")
