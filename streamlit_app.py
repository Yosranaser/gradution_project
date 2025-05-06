import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
st.set_page_config(page_title="Smart Car Assistant", layout="centered")

st.title("🚗Your Smart Car Assistant")
st.subheader("مرحبًا بك!")

st.markdown("""
### 👋 إزّيك؟  
الموقع ده معمول علشان يساعدك تعرف حالة عربيتك بسهولة.

- لو العربية فيها مشكلة أو محتاجة صيانة، هنقولك فورًا.
- تقدر كمان تشوف **نسبة البنزين**، **السرعة**، و**درجة حرارة الأجزاء المهمة** وانت سايق أو قبل ما تتحرك.

كل اللي عليك:
- تقول **"صيانة"** لو حابب تتطمن على حالة العربية.
- أو تقول **"عرض البيانات"** لو حابب تشوف كل حاجة شغالة إزاي دلوقتي.

اختار واحدة من تحت 👇
""")

col1, col2 = st.columns(2)
with col1:
    dashboard = st.button("👁️ عرض البيانات")
with col2:
    maintenance = st.button("🛠️ صيانة")

cred = credentials.Certificate("path/to/your/firebase_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://your-project.firebaseio.com/'
})

# Read data from Firebase
ref = db.reference('car_status')
data = ref.get()

fuel = data['fuel_level']
speed = data['speed']
temp = data['engine_temperature']
# عرض محتوى حسب الاختيار
if dashboard:
    st.success("هندخلك على عرض البيانات...")
    st.write("هنا هنوريك البنزين، السرعة، الفولت، ودرجة الحرارة.")

elif maintenance:
    st.success("هندخلك على صفحة الصيانة...")
    st.write("هنا هنعرض لك حالة كل جزء في العربية، وهل محتاج يتصلح ولا تمام.")
