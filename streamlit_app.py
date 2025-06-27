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



