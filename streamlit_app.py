import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
from PIL import Image
import io
import requests
import plotly.graph_objects as go
st.set_page_config(layout="wide")
st.sidebar.title("🚗 Car App Navigation")
page = st.sidebar.selectbox("اختر الصفحة:", ["الصفحة الرئيسية", "Dashboard"])
if page == "Dashboard":
   st.title("🚗 Dashboard")
   st.write("هنا يتم عرض الداشبورد والبيانات الخاصة بحالة السيارة.")
   st.set_page_config(page_title="Dashboard", page_icon="🚗", layout="wide")
   st.title("🚗 Car Health Dashboard")
   url ="https://raw.githubusercontent.com/Yosranaser/gradution_project/refs/heads/main/predictive_maintenance_final_fixed.csv"
   data = pd.read_csv(url)
   
   st.subheader("🔧 حالة المكونات الرئيسية")
   
   # 🔥 تقسيم الصفحة إلى 3 أعمدة
   col1, col2, col3 = st.columns(3)
   
   col1.metric("ESP32 Temp (°C)", f"{data['esp32_temperature_(°c)']} °C")
   col2.metric("Servo Temp (°C)", f"{data['servo_temperature_(°c)']} °C")
   col3.metric("Motor Driver Temp (°C)", f"{data['motor_driver_temperature_(°c)']} °C")
   
   col1.metric("STM32 Voltage (V)", f"{data['stm32_voltage_(v)']} V")
   col2.metric("Universal Voltage (V)", f"{data['universal_voltage_(v)']} V")
   col3.metric("Servo Vibration (g)", f"{data['servo_vibration_(g)']} g")
   
   
   if data["ultrasonic_signal_loss"].iloc[-1] > 0:
       st.error(f"🚨 Ultrasonic Signal Loss Detected: {data['ultrasonic_signal_loss']}")
   else:
       st.success("✅ No Signal Loss Detected in Ultrasonic Sensor")
   servo_temp = data['servo_temperature_(°c)'].iloc[-1]
  
   fig = go.Figure(go.Indicator(
   mode="gauge+number+delta",
   value=servo_temp,  # تأكد أن هذا المتغير رقم (int أو float)
   delta={'reference': 70},
   title={'text': "Servo Temperature (°C)"},
   gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 60], 'color': "lightgreen"},
            {'range': [60, 85], 'color': "orange"},
            {'range': [85, 100], 'color': "red"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 85
        }
    }
))


   
   st.plotly_chart(fig, use_container_width=True)
   
   
   st.subheader("📜 Summary Data")
   
   data_table = pd.DataFrame(list(data.items()), columns=["Component", "Value"])
   st.dataframe(data_table)                
elif page=="الصفحة الرئيسية"
   col1, col2 = st.columns([1,1])
   with col1:
       st.markdown("""
       <div style="background-color:#f2f2f2; padding:20px; border-radius:15px; direction:rtl; text-align:right;">
           <p style="font-size:20px; color:#1f77b4;">
               الموقع ده معمول علشان <b>يطمنك على حالة عربيتك</b>.
               <span style="color:#d62728;">بس قبل ما نقولك المشكلة أو لو فيها حاجة، لازم الأول نتأكد إنك صاحب العربية عن طريق صورة ليك.</span>
           </p>
   
           
               ✔️ لو العربية فيها أي مشكلة أو محتاجة صيانة، هنقولك فورًا.
               ✔️ وكمان تقدر تعرف <b>نسبة البنزين</b>، <b>السرعة</b>، و<b>درجة حرارة الأجزاء المهمة</b>.
           </p>
   
          
               يعني الموضوع بسيط جدًا:
               <br>➡️ <b>خد صورة ليك الأول</b> علشان نتأكد إنك صاحب العربية.
               <br>وبعدين نبدأ نشوف حالة العربية مع بعض.
   
   
   
   
   
   
   
   
               
           </p>
       </div>
       """, unsafe_allow_html=True)
   with col2:
        st.image("WhatsApp Image 2025-06-28 at 13.41.48_b3ecfd63.jpg", caption="your car")
   
   
   ref1 = cv2.imread("yossra.jpg", 0)
   ref2 = cv2.imread("shorouk2.jpg", 0)
   flag=0
   col1, col2 = st.columns([1,1])
   with col1:
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
           with  col2 :
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
        
        
