# utils.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data(filename='german_credit_data.csv', data_folder='data'):
    """
    Tự động tìm và đọc file dữ liệu csv từ các thư mục lân cận.
    """
    current_dir = os.getcwd()
    possible_paths = [
        os.path.join(current_dir, data_folder, filename),
        os.path.join(current_dir, '..', data_folder, filename),
        os.path.join(current_dir, '..', '..', data_folder, filename)
    ]
    
    for path in possible_paths:
        full_path = os.path.normpath(path)
        if os.path.exists(full_path):
            print(f"Đã tìm thấy file tại: {full_path}")
            return pd.read_csv(full_path)
            
    raise FileNotFoundError(f"Không tìm thấy file {filename} trong các thư mục dự kiến.")

def plot_distributions(df, cols=["Age", "Credit amount", "Duration"]):
    """Vẽ biểu đồ phân phối (Histogram) cho các cột số."""
    df[cols].hist(bins=7, edgecolor="black", figsize=(10, 6))
    plt.suptitle("Phân phối của các đặc trưng số", fontsize=14)
    plt.show()

def plot_boxplots(df, cols=["Age", "Credit amount", "Duration"]):
    """Vẽ biểu đồ Boxplot để phát hiện Outliers."""
    plt.figure(figsize=(10, 5))
    for i, col in enumerate(cols):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(y=df[col], color="skyblue")
        plt.title(col)
    plt.tight_layout()
    plt.show()
    

def save_model(model, model_name, sub_folder='models'):
    """
    Lưu model vào thư mục data/[sub_folder].
    Ví dụ: data/models/RandomForest.pkl
    """
    # 1. Xác định vị trí thư mục data hiện tại
    current_dir = os.getcwd()
    
    # Logic tìm thư mục data (tương tự hàm load_data)
    data_path = os.path.join(current_dir, 'data')
    if not os.path.exists(data_path):
        # Thử lùi lại 1 cấp nếu không thấy
        data_path = os.path.join(current_dir, '..', 'data')
    
    # 2. Tạo đường dẫn tới thư mục lưu model (VD: data/models)
    save_dir = os.path.join(data_path, sub_folder)
    
    # 3. Tạo thư mục nếu chưa tồn tại (quan trọng!)
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. Tạo đường dẫn file đầy đủ
    # Xử lý tên file để bỏ các ký tự lạ nếu có (VD: khoảng trắng -> _)
    clean_name = model_name.replace(" ", "_") + ".pkl"
    file_path = os.path.join(save_dir, clean_name)
    
    # 5. Lưu model
    joblib.dump(model, file_path)
    print(f"💾 Đã lưu model tại: {file_path}")    