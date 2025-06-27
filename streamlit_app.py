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

uploaded_file = st.file_uploader("🗂️ ارفع ملف CSV للبيانات", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=";")

        # ✅ تصحيح الأعمدة
        df.columns = df.columns.str.replace('�', '°')
        df.columns = df.columns.str.replace('(?c)', '(°c)', regex=False)

        st.success("✅ تم رفع البيانات بنجاح")
        st.dataframe(df)

        # ✅ تحميل الموديل
        with open('model (7).pkl', 'rb') as file:
            model = pickle.load(file)

        expected_features = list(model.feature_names_in_)

        # ✅ تحقق من وجود الأعمدة
        missing = [col for col in expected_features if col not in df.columns]
        if missing:
            st.error(f"❌ الأعمدة الناقصة: {missing}")
        else:
            selected_df = df[expected_features]

            # ✅ التنبؤ
            prediction = model.predict(selected_df)[0]
            st.subheader(f"⚙️ Prediction Result: **{prediction}**")
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

    except Exception as e:
        st.error(f"❌ حصل خطأ: {e}")
else:
    st.warning("⚠️ من فضلك ارفع ملف CSV قبل التنبؤ.")


