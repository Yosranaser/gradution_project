import streamlit as st
import os
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
st.set_page_config(page_title="Smart Car Assistant", layout="centered")

st.title("🚗Your Smart Car Assistant")
st.subheader("مرحبًا بك!")

st.markdown("""
### 👋 إزّيك؟  
الموقع ده معمول علشان يساعدك تعرف حالة عربيتك بسهولة.

- لو العربية فيها مشكلة أو محتاجة صيانة، هنقولك فورًا.
- تقدر كمان تشوف *نسبة البنزين، **السرعة، ودرجة حرارة الأجزاء المهمة* وانت سايق أو قبل ما تتحرك.

كل اللي عليك:
- تقول *"صيانة"* لو حابب تتطمن على حالة العربية.
- أو تقول *"عرض البيانات"* لو حابب تشوف كل حاجة شغالة إزاي دلوقتي.

اختار واحدة من تحت 👇
""")

col1, col2 = st.columns(2)
with col1:
    dashboard = st.button("👁 عرض البيانات")
with col2:
    maintenance = st.button("🛠 صيانة")

cred = credentials.Certificate("predictive-maintance-data-firebase-adminsdk-fbsvc-e6efdfda3e.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://predictive-maintance-data-default-rtdb.firebaseio.com/'
})
# Read data from Firebase
fuel = db.reference('fuel_level').get()
speed = db.reference('speed').get()
temp = db.reference('engine_temperature').get()
# عرض محتوى حسب الاختيار
if dashboard:
    st.success("هندخلك على عرض البيانات...")
    st.write("هنا هنوريك البنزين، السرعة، الفولت، ودرجة الحرارة.")
    st.metric(label="🚀 السرعة", value=f"{speed} كم/س")
    st.metric(label="⛽ نسبة البنزين", value=f"{fuel}%")
    st.metric(label="🌡 درجة حرارة المحرك", value=f"{temp}°C")

    if temp > 100:
        st.error("⚠ درجة حرارة المحرك عالية جدًا! راجع الفني فورًا.")

elif maintenance:
    st.success("هندخلك على صفحة الصيانة...")
    st.write("هنا هنعرض لك حالة كل جزء في العربية، وهل محتاج يتصلح ولا تمام.")
