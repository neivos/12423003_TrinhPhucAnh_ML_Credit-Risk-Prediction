import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Credit Risk Scoring", layout="centered")
st.title("💳 Dự báo rủi ro tín dụng")

@st.cache_resource
def load_resources():
    # --- THAY ĐỔI: Tìm folder data/pkl ---
    base_dir = os.getcwd()
    
    # 1. Xác định vị trí folder data/pkl
    paths_to_check = [
        os.path.join(base_dir, 'data', 'pkl'),       
        os.path.join(base_dir, '..', 'data', 'pkl') 
    ]
    
    model_dir = None
    for p in paths_to_check:
        if os.path.exists(p):
            model_dir = p
            break
            
    if not model_dir:
        st.error("❌ Không tìm thấy thư mục 'data/pkl'. Hãy chạy train_model.py trước.")
        return None, None

    # Load Models
    models = {}
    model_files = ["XGBoost", "Random_Forest", "Decision_Tree", "Logistic_Regression"]
    
    for name in model_files:
        path = os.path.join(model_dir, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
            
    # Load Encoders
    encoders = {}
    enc_cols = ["Sex", "Housing", "Saving accounts", "Checking account"]
    
    for col in enc_cols:
        path = os.path.join(model_dir, f"{col}_encoder.pkl")
        if os.path.exists(path):
            encoders[col] = joblib.load(path)
            
    return models, encoders

models, encoders = load_resources()

if not models or not encoders:
    st.stop()

# --- PHẦN DƯỚI GIỮ NGUYÊN ---
st.sidebar.header("Cấu hình")
selected_model_name = st.sidebar.selectbox("Chọn mô hình", list(models.keys()))
model = models[selected_model_name]

with st.form("input_form"):
    st.subheader("Thông tin khách hàng")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Tuổi", 18, 80, 30)
        sex = st.selectbox("Giới tính", encoders["Sex"].classes_)
        job = st.selectbox("Công việc", [0, 1, 2, 3])
        housing = st.selectbox("Nhà ở", encoders["Housing"].classes_)
        
    with col2:
        saving = st.selectbox("TK Tiết kiệm", encoders["Saving accounts"].classes_)
        checking = st.selectbox("TK Vãng lai", encoders["Checking account"].classes_)
        credit_amount = st.number_input("Số tiền vay", 0, 20000, 1000)
        duration = st.number_input("Kỳ hạn (tháng)", 1, 72, 12)
    
    submit = st.form_submit_button("Dự báo")

if submit:
    try:
        sex_enc = encoders["Sex"].transform([sex])[0]
        housing_enc = encoders["Housing"].transform([housing])[0]
        saving_enc = encoders["Saving accounts"].transform([saving])[0]
        checking_enc = encoders["Checking account"].transform([checking])[0]
        
        input_data = pd.DataFrame([[
            age, sex_enc, job, housing_enc, saving_enc, checking_enc, credit_amount, duration
        ]], columns=["Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Credit amount", "Duration"])
        
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else 0
        
        st.divider()
        if pred == 1:
            st.error(f"⚠️ Rủi ro cao (Bad Credit) - Xác suất: {proba:.2%}")
        else:
            st.success(f"✅ An toàn (Good Credit) - Xác suất rủi ro: {proba:.2%}")
            
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")