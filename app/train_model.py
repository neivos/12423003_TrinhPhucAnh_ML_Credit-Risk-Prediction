import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

import utils 

def main():
    # --- 1. Load dữ liệu ---
    print("🚀 Đang tải dữ liệu...")
    try:
        df = utils.load_data()
    except FileNotFoundError as e:
        print(f"❌ Lỗi: {e}")
        return

    # --- 2. Tiền xử lý ---
    print("🛠 Đang xử lý dữ liệu...")
    
    df['Saving accounts'] = df['Saving accounts'].fillna('Unknown')
    df['Checking account'] = df['Checking account'].fillna('Unknown')

    features = ["Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Credit amount", "Duration"]
    target = "Risk"
    
    df_model = df[features + [target]].copy()
    df_model[target] = df_model[target].map({'bad': 1, 'good': 0})

    cat_cols = ["Sex", "Housing", "Saving accounts", "Checking account"]
    
    # --- THAY ĐỔI: Tạo folder 'pkl' trong 'data' ---
    current_dir = os.getcwd()
    
    # 1. Tìm folder data gốc trước
    data_dir = os.path.join(current_dir, 'data')
    if not os.path.exists(data_dir):
        data_dir = os.path.join(current_dir, '..', 'data')
        
    # 2. Tạo folder con 'pkl'
    save_dir = os.path.join(data_dir, 'pkl')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"📂 Các file model sẽ được lưu tại: {save_dir}")

    # Encode và lưu Encoders
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        joblib.dump(le, os.path.join(save_dir, f"{col}_encoder.pkl"))

    # --- 3. Chia tập & Train ---
    X = df_model.drop(columns=[target])
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Decision_Tree": DecisionTreeClassifier(random_state=42),
        "Random_Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    print("🔥 Đang huấn luyện...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"   -> {name:20} : AUC = {auc:.4f}")
        
        joblib.dump(model, os.path.join(save_dir, f"{name}.pkl"))

    print("\n✅ HUẤN LUYỆN XONG! File đã lưu vào 'data/pkl'.")
    print("👉 Hãy chạy: streamlit run app.py")

if __name__ == "__main__":
    main()