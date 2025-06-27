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
import pandas as pd
import xgboost as xgb
import joblib
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

sheet_id = "10GFBlxh8nNU-yIe7_UH0O6UDqW4Uv_fc0zNR_xC_O00"
sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
df = pd.read_csv(sheet_url)
st.title("📊 بيانات السيارة من Google Sheet")
st.dataframe(df)


# model = xgb.XGBClassifier()
# model.load_model('model.json')
with open('model (2).pkl', 'rb') as f:
    model = pickle.load(f)

        # ✅ التنبؤ
if st.button("🔍 Predict Car Status"):
   prediction = model.predict(df)[0]
   st.subheader(f"⚙️ Prediction Result: **{prediction}**")


           

    
