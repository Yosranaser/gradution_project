import streamlit as st
import os
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
import streamlit as st
import cv2
import io
import pickle

with open('model (2).pkl', 'rb') as f:
    model = pickle.load(f)
st.set_page_config(layout="wide")
col1, col2 = st.columns([2, 1])  # col1 = يسار، col2 = يمين
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


   
    
if not firebase_admin._apps:
    cred = credentials.Certificate('predictive-maintance-data-firebase-adminsdk-fbsvc-e6efdfda3e.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://predictive-maintance-data-default-rtdb.firebaseio.com/'
    })
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
 
prediction = model.predict(df)[0]
st.header("🔍 نتيجة التحليل")
if prediction == 1:
    st.error("🚨 النظام يتوقع وجود مشكلة أو عطل! راجع الفني فورًا.")
else:
    st.success("✅ كل القيم طبيعية، لا توجد مؤشرات على الأعطال.")
st.subheader("📊 البيانات الحالية:")
st.dataframe(df.T.rename(columns={0: "القيمة"}))









