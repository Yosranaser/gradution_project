import streamlit as st
import os
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
import streamlit as st
import cv2
import io

import streamlit as st

# إعداد العرض على اليمين
st.set_page_config(layout="wide")

# تقسيم الصفحة: عمودين، النص في اليمين
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


   
    
cred = credentials.Certificate("predictive-maintance-data-firebase-adminsdk-fbsvc-35435ce836.json")
firebase_admin.initialize_app(cred, {
'databaseURL': 'https://predictive-maintance-data-default-rtdb.firebaseio.com/' 
})
    
# Read data from Firebase
fuel = db.reference('fuel_level').get()
speed = db.reference('speed').get()
temp = db.reference('engine_temperature').get()  
if flag==1 :    
    col1, col2 = st.columns(2)
    with col1:
        dashboard = st.button("👁 عرض البيانات")
    with col2:
        maintenance = st.button("🛠 صيانة")
    

    
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
