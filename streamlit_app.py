import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
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
ref1 = cv2.imread("yossra.jpg", 0)
ref2 = cv2.imread("shorouk2.jpg", 0)
flag=0
uploaded_image = st.camera_input("Take your picture")
if uploaded_image is not None:
    user_img = Image.open(io.BytesIO(uploaded_image.read())).convert("L")
    user_img_np = np.array(user_img)

    
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(user_img_np, None)
    kp2, des2 = orb.detectAndCompute(ref1, None)
    kp3, des3 = orb.detectAndCompute(ref2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches1 = bf.match(des1, des2)
    matches2 = bf.match(des1, des3)

    score1 = len(matches1)
    score2 = len(matches2)

    if score1 > score2 and score1 > 20:
        st.success("✅ Face matched with yossra ")
        st.image("yossra.jpg", caption="yossra naser has sussessifully logged in")
        flag=1
    elif score2 > score1 and score2 > 20:
        flag=1
        st.success("✅ Face matched with shorouk")
        st.image("shorouk2.jpg", caption="shorouk ahmed has sussessifully logged in ")
        
    else:
        st.error("❌ Face not recognized")
        flag=0
        cap.release()       
        cv2.destroyAllWindows()
if flag==1:
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
    
    
