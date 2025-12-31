# preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    """Làm sạch dữ liệu cơ bản: bỏ cột index thừa, điền giá trị thiếu."""
    # Bỏ cột Unnamed: 0 nếu tồn tại
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    
    # Xử lý giá trị thiếu
    df['Saving accounts'] = df['Saving accounts'].fillna('Unknown')
    df['Checking account'] = df['Checking account'].fillna('Unknown')
    
    return df

def encode_and_split(df, target_col='Risk', test_size=0.2, random_state=42):
    """
    Mã hóa dữ liệu (One-hot cho features, Label cho target) 
    và chia tập train/test.
    """
    # Xác định các cột phân loại (categorical)
    categorical_cols = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
    
    # Mã hóa biến Target (Risk: good/bad -> 1/0)
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    
    # Mã hóa One-Hot cho các biến input phân loại
    # drop_first=True để tránh đa cộng tuyến (dummy variable trap)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Tách X và y
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test