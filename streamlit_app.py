import streamlit as st
import os
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2
import io

st.title("Face Authentication using DeepFace")

# تحميل الصور المخزنة
ref1 = cv2.imread("yossra.jpg")
ref2 = cv2.imread("shorouk.jpg")

uploaded_image = st.camera_input("Take a picture")

if uploaded_image is not None:
    img_bytes = uploaded_image.getvalue()
    img = Image.open(io.BytesIO(img_bytes))
    img_np = np.array(img)

    # قارن مع الصورتين
    result = None
    try:
        result = DeepFace.verify(img_np, ref1, enforce_detection=False)
        if result["verified"]:
            st.success("✅ Face matched with Person 1")
            st.image(ref1, caption="Person 1")
        else:
            result = DeepFace.verify(img_np, ref2, enforce_detection=False)
            if result["verified"]:
                st.success("✅ Face matched with Person 2")
                st.image(ref2, caption="Person 2")
            else:
                st.error("❌ Face not recognized")
    except Exception as e:
        st.error(f"Error during verification: {e}")


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
import cv2
import streamlit as st
import numpy as np

st.title("Face Detection")

uploaded_image = st.camera_input("Take a picture")

if uploaded_image is not None:
    img = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(img, 1)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    st.image(img, channels="BGR")

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
