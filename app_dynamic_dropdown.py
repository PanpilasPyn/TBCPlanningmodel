
import streamlit as st
import pandas as pd
import joblib

# โหลดโมเดลและ columns
model = joblib.load("rf_model_colab_ui.pkl")
columns = joblib.load("feature_columns_colab_ui.pkl")

st.set_page_config(page_title="Good Speed Predictor", layout="centered")
st.title("🚀 พยากรณ์ Good Speed run (Can Per Hour)")

# 🔍 สร้าง dropdown จาก .pkl (ชื่อฟีเจอร์ที่ใช้ one-hot)
def extract_categories(prefix):
    return sorted(set(c.replace(prefix + "_", "") for c in columns if c.startswith(prefix + "_")))

drink_types = extract_categories("Drink Type")
customers = extract_categories("Customer")
design_types = extract_categories("Design type")
coil_types = extract_categories("Coil type")
can_sizes = extract_categories("Can Size")

# ฟอร์มรับข้อมูล
with st.form("predict_form"):
    drink_type = st.selectbox("Drink Type", drink_types)
    good_qty = st.number_input("Good Qty (Can)", value=500000)
    customer = st.selectbox("Customer", customers)
    design_type = st.selectbox("Design Type", design_types)
    avg_speed_before = st.number_input("Average Speed Month Before", value=48000)
    coil_type = st.selectbox("Coil Type", coil_types)
    can_size = st.selectbox("Can Size", can_sizes)

    submitted = st.form_submit_button("พยากรณ์")

if submitted:
    # สร้าง dataframe จาก input
    input_dict = {
        "Drink Type": drink_type,
        "Good Qty (Can)": good_qty,
        "Customer": customer,
        "Design type": design_type,
        "Average speed month before": avg_speed_before,
        "Coil type": coil_type,
        "Can Size": can_size
    }

    df = pd.DataFrame([input_dict])
    for col in ["Drink Type", "Customer", "Design type", "Coil type", "Can Size"]:
        df[col] = df[col].astype(str)
    df_encoded = pd.get_dummies(df)

    for col in columns:
        if col not in df_encoded:
            df_encoded[col] = 0
    df_encoded = df_encoded[columns]

    # ทำนาย
    prediction = model.predict(df_encoded)[0]
    st.success(f"📈 พยากรณ์ความเร็ว Good Speed run = {prediction:,.2f} cans/hour")
