import streamlit as st
import os
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
import streamlit as st
import cv2
import io

st.title("Face Authentication using DeepFace")

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
        st.image("yossra.jpg", caption="yossra naser")
    elif score2 > score1 and score2 > 20:
        st.success("✅ Face matched with shorouk")
        st.image("shorouk2.jpg", caption="shorouk ahmed")
        cap.release()       
        
        flag=1
    else:
        st.error("❌ Face not recognized")
        flag=0
        cap.release()       
        cv2.destroyAllWindows()

if flag==1 :
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
    
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("predictive-maintance-data-firebase-adminsdk-fbsvc-e6efdfda3e.json")
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://predictive-maintance-data-default-rtdb.firebaseio.com/'
            })
    
        # محاولة قراءة بيانات
        fuel = db.reference('fuel_level').get()
        print("Fuel Level:", fuel)
    
    except RefreshError as e:
        st.error("فشل في المصادقة مع Google. تأكد من اتصال الإنترنت وصحة ملف الخدمة.")
        st.stop()
    except Exception as e:
        st.error(f"حدث خطأ آخر: {e}")
        st.stop()
    
    # Read data from Firebase
    fuel = db.reference('fuel_level').get()
    speed = db.reference('speed').get()
    temp = db.reference('engine_temperature').get()
    
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
