import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
from PIL import Image
import io
import requests
import plotly.graph_objects as go
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from streamlit_js_eval import streamlit_js_eval
import requests
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.models import load_model
def parse_hex_line(line):
    line = line.strip()
    if not line.startswith(':') or len(line) < 11:
        return None
    try:
        byte_count = int(line[1:3], 16)
        address = int(line[3:7], 16)
        record_type = int(line[7:9], 16)
        data = [int(line[i:i+2], 16) for i in range(9, 9 + byte_count * 2, 2)]
        checksum = int(line[9 + byte_count * 2: 9 + byte_count * 2 + 2], 16)


        total = byte_count + (address >> 8) + (address & 0xFF) + record_type + sum(data)
        total = (total & 0xFF)
        calc_checksum = ((~total + 1) & 0xFF)
        is_valid = (checksum == calc_checksum)


        repeated = int(len(set(data)) == 1 and len(data) > 0)
        entropy = calculate_entropy(data)
        max_val = max(data) if data else 0
        min_val = min(data) if data else 0
        is_ff_pattern = int(max_val == 255 and min_val == 255)

        return {
            "byte_count": byte_count,
            "address": address,
            "record_type": record_type,
            "data_bytes": data,
            "checksum": checksum,
            "valid_checksum": int(is_valid),
            "repeated_pattern": repeated,
            "entropy": entropy,
            "max_val": max_val,
            "min_val": min_val,
            "is_ff_pattern": is_ff_pattern
        }
    except:
        return None

def hex_file_to_dataframe(file_path):
    rows = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            parsed = parse_hex_line(line)
            if parsed:
                parsed["line_index"] = idx
                rows.append(parsed)
    df = pd.DataFrame(rows)
    return df
def pad_data(data_list, max_len=16):
    if len(data_list) > max_len:
        return data_list[:max_len]
    else:
        return data_list + [0] * (max_len - len(data_list))


def predict_hex_file(model_path, uploaded_file, max_lines=100, max_data_len=16):
    # 1. قراءة محتوى الملف
    hex_content = uploaded_file.read().decode("utf-8")
    lines = hex_content.strip().splitlines()

    # 2. تحويله إلى DataFrame
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hex", mode="w") as tmp:
        tmp.write(hex_content)
        tmp_path = tmp.name

    df = hex_file_to_dataframe(tmp_path)

    # 3. تجهيز الخصائص بنفس شكل التدريب
    feature_rows = []
    for _, row in df.head(max_lines).iterrows():
        data = pad_data(row["data_bytes"], max_data_len)
        features = data + [
            row["record_type"],
            row["valid_checksum"],
            row["repeated_pattern"]
        ]
        feature_rows.append(features)

    # تكملة لو عدد الأسطر أقل من 100
    while len(feature_rows) < max_lines:
        feature_rows.append([0] * (max_data_len + 3))

    input_data = np.expand_dims(np.array(feature_rows), axis=0)  # شكل (1, 100, 19)

    # 4. تحميل الموديل والتنبؤ
    model = load_model(model_path)
    prediction = model.predict(input_data)[0][0]

    return prediction


def get_location_by_ip():
    url = "https://ipinfo.io/json"
    response = requests.get(url)
    data = response.json()
    loc = data['loc'].split(',')
    latitude = float(loc[0])
    longitude = float(loc[1])
    return latitude, longitude


def find_place_osm(query, lat, lon):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': query,
        'format': 'json',
        'limit': 5,
        'addressdetails': 1,
        'viewbox': f"{lon-0.05},{lat-0.05},{lon+0.05},{lat+0.05}",
        'bounded': 1
    }

    headers = {
        'User-Agent': 'SmartCarApp/1.0 (yosranaser43@gmail.com)'  
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()  # لو في خطأ HTTP يوقف
        data = response.json()
    except requests.exceptions.HTTPError as errh:
        st.error(f"HTTP Error: {errh}")
        return []
    except requests.exceptions.ConnectionError as errc:
        st.error(f"Error Connecting: {errc}")
        return []
    except requests.exceptions.Timeout as errt:
        st.error(f"Timeout Error: {errt}")
        return []
    except requests.exceptions.RequestException as err:
        st.error(f"OOps: Something Else {err}")
        return []
    except Exception as e:
        st.error(f"Unknown error: {e}")
        st.error(response.text)
        return []

    if data:
        results = []
        for place in data:
            name = place.get("display_name")
            lat = place.get("lat")
            lon = place.get("lon")
            results.append(f"📍 {name} (Lat: {lat}, Lon: {lon})")
        return results
    else:
        return ["🚫 لم يتم العثور على مكان مطابق."]

def generate_response(intent):
    if intent == "nearest_gas":
        return "🛢️ أقرب محطة بنزين هي محطة وطنية على بعد 2.3 كم."
    elif intent == "nearest_restaurant":
        return "🍽️ أقرب مطعم هو مطعم البركة على بعد 1.5 كم."
    elif intent == "traffic_info":
        return "🚦 الطريق حالياً مزدحم قليلاً عند شارع الجامعة."
    elif intent == "navigate":
        return "🗺️ جاري فتح الاتجاهات إلى وجهتك."
    else:
        return "🤖 لم أفهم سؤالك تماماً، حاول بصيغة أخرى."
def detect_intent(user_input):
    if "محطة بنزين" in user_input:
        return "nearest_gas"
    elif "مطعم" in user_input:
        return "nearest_restaurant"
    elif "زحمة" in user_input or "الطريق" in user_input:
        return "traffic_info"
    elif "اوصل" in user_input or "اتجه" in user_input:
        return "navigate"
    else:
        return "general"
st.set_page_config(layout="wide")
st.sidebar.title("🚗 Car App Navigation")
page = st.sidebar.selectbox("اختر الصفحة:", ["الصفحة الرئيسية", "Dashboard","chatbot","hex file attack detection"])

#-----------------------------------------------------------------------------------------------
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
   
   st.subheader("📋 Key Features Overview")

  
   cols_per_row = 3

   for i in range(0, len(data.columns), cols_per_row):
       cols = st.columns(cols_per_row)
       for idx, col in enumerate(data.columns[i:i+cols_per_row]):
           data[col] = pd.to_numeric(data[col], errors='coerce')
           with cols[idx]:
               st.metric(
                   label=col,
                   value=f"{data[col].mean():.2f}"
               )
   for col in data.columns:
    st.markdown(f"### 📌 {col} Distribution")

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=data[col],
        nbinsx=20,
        name="Distribution",
        marker_color='lightblue',
        opacity=0.75
    ))

    # ✅ إضافة خطوط حدود — عدل حسب الفيتشر
    min_value = data[col].min()
    max_value = data[col].max()

    # مثال على تقسيم المستويات (يمكن تخصيصه حسب الفيتشر):
    normal_max = data[col].mean()
    warning_max = normal_max + (max_value - normal_max) * 0.5
    danger_min = warning_max

    # ✅ مناطق Normal
    fig.add_vrect(
        x0=min_value, x1=normal_max,
        fillcolor="green", opacity=0.2,
        line_width=0,
        annotation_text="Normal"
    )

    # ✅ مناطق Warning
    fig.add_vrect(
        x0=normal_max, x1=warning_max,
        fillcolor="orange", opacity=0.2,
        line_width=0,
        annotation_text="Warning"
    )

    # ✅ مناطق Danger
    fig.add_vrect(
        x0=danger_min, x1=max_value,
        fillcolor="red", opacity=0.2,
        line_width=0,
        annotation_text="Danger"
    )

    # إعدادات الشكل العام
    fig.update_layout(
        title=f"{col} Distribution Histogram",
        xaxis_title=col,
        yaxis_title="Count",
        bargap=0.05,
        template="plotly_white"
    )

    # عرض الرسم
    st.plotly_chart(fig)
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
#-------------------------------------------------------------------------------
elif page=="chatbot":
    #geolocator = Nominatim(user_agent="smartcar-app")
    #reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
  
    latitude, longitude =30.059556,31.226320
    st.write(f"📍 خط العرض: {latitude}, خط الطول: {longitude}")

   
    
    
    place_type = st.selectbox(
        "🔍 اختر نوع المكان اللي بتدور عليه:",
        {
            "محطة بنزين": {"amenity": "fuel"},
            "مطعم": {"amenity": "restaurant"},
            "صيدلية": {"amenity": "pharmacy"},
            "موقف سيارات": {"amenity": "parking"},
            "مستشفى": {"amenity": "hospital"}
        }.keys()
    )
    
    # ✅ إعداد التاج للبحث
    tags_dict = {
        "محطة بنزين": {"amenity": "fuel"},
        "مطعم": {"amenity": "restaurant"},
        "صيدلية": {"amenity": "pharmacy"},
        "موقف سيارات": {"amenity": "parking"},
        "مستشفى": {"amenity": "hospital"}
    }
    
    tags = tags_dict[place_type]
    
    # ✅ زر البحث
    if st.button("🔍 ابحث عن أقرب مكان"):
        try:
            with st.spinner("جاري البحث..."):
                # ✅ البحث باستخدام OSMnx
                gdf = ox.features.features_from_point(
                    (latitude, longitude), tags=tags, dist=2000
                )
    
                if not gdf.empty:
                    st.success(f"✅ تم العثور على {len(gdf)} {place_type}(s) في نطاق 2 كم:")
                    results = []
                    for index, row in gdf.iterrows():
                        name = row.get('name', '📍 مكان بدون اسم')
                        place_lat = row.geometry.centroid.y
                        place_lon = row.geometry.centroid.x
    
                        # ✅ جلب العنوان باستخدام reverse geocoding
                        try:
                            location = reverse((place_lat, place_lon))
                            address = location.address if location else "🚫 عنوان غير متوفر"
                        except:
                            address = "🚫 عنوان غير متوفر"
    
                        results.append({
                            '📍 الاسم': name,
                            '🏠 العنوان': address,
                            '📍 خط العرض': place_lat,
                            '📍 خط الطول': place_lon
                        })
                    df = pd.DataFrame(results)
                    st.dataframe(df)
    
                    # ✅ رسم خريطة تفاعلية
                    st.map(df.rename(columns={"📍 خط العرض": "lat", "📍 خط الطول": "lon"}))
                else:
                    st.warning("🚫 لم يتم العثور على أماكن قريبة في هذا النطاق.")
        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء البحث: {e}")

#------------------------------------------------------------------------
elif page=="الصفحة الرئيسية":
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
                   st.success("✅ Face matched with yossra ")
                   st.image("yossra.jpg", caption="yossra naser has sussessifully logged in")
                   flag=1
                   
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

  
       
elif page=="hex file attack detection":
    st.title("🔐 HEX File Attack Detection")
    st.write(""" قبل تثبيت أي تحديث جديد للـ Firmware،
    يتم تحليل الملف تلقائيًا باستخدام نموذج ذكاء صناعي للكشف عن أي نمط هجوم أو تعديل غير مصرح به. هذه الخطوة تضمن سلامة النظام وتحميه من التهديدات المحتملة قبل أن يتم تحميل التحديث على وحدة التحكم.""")
    uploaded_file = st.file_uploader("📂 ارفع ملف HEX لفحصه", type=["hex"])
    
    if uploaded_file is not None:
        st.info("📊 جاري التحليل...")
    
        prediction = predict_hex_file("hex_model.h5", uploaded_file)
    
        if prediction >= 0.5:
            st.error(f"🚨 الملف يحتمل أن يكون **مصاب** بهجوم. (Confidence: {prediction:.2f})")
        else:
            st.success(f"✅ الملف يبدو **سليماً**. (Confidence: {prediction:.2f})")


        
