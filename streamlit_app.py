import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import streamlit as st
import face_recognition
import cv2
import os
import numpy as np
from PIL import Image
known_names = ["yossra", "shorouk"]
known_encodings = []

for name in known_names:
    image_path = f"{name}.jpg"
    if os.path.exists(image_path):
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_encodings.append(encoding[0])
        else:
            st.warning(f"⚠️ الصورة {name}.jpg لم يتم العثور على وجه فيها")
    else:
        st.warning(f"⚠️ الصورة {name}.jpg غير موجودة")

# -------------------------------
# واجهة Streamlit
# -------------------------------
st.title("🎥 Face Authentication")
st.write("من فضلك فعّل الكاميرا لتسجيل دخولك...")

# زر تشغيل الكاميرا
if st.button("ابدأ التحقق"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    result = None
    name_detected = None

    for _ in range(100):  # نحاول كذا إطار للتعرف
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding)
            face_distances = face_recognition.face_distance(known_encodings, encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name_detected = known_names[best_match_index]
                result = True
                break

        stframe.image(rgb_frame, channels="RGB")

        if result:
            break

    cap.release()
    cv2.destroyAllWindows()

    if result:
        st.success(f"مرحبًا {name_detected.capitalize()} ✅")
        image_path = f"{name_detected}.jpg"
        st.image(image_path, caption=name_detected)
        # هنا تكملي فتح الموقع أو باقي المميزات
        st.markdown("### ✅ تم الدخول بنجاح، جاري فتح الموقع ...")
    else:
        st.error("❌ آسف، لم يتم التعرف على الوجه. لا يمكنك الدخول.")

