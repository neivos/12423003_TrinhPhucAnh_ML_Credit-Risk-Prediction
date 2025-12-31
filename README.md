# DỰ ĐOÁN RỦI RO TÍN DỤNG KHÁCH HÀNG BẰNG MACHINE LEARNING

---

## 1. Giới thiệu đề tài

### 🔹 Bài toán
Trong lĩnh vực tài chính – ngân hàng, việc đánh giá **rủi ro tín dụng của khách hàng** đóng vai trò rất quan trọng trong quá trình ra quyết định cho vay.  
Bài toán đặt ra là: **dựa trên thông tin tài chính và hành vi của khách hàng, dự đoán khả năng khách hàng thuộc nhóm rủi ro tốt hay xấu**.

Đây là một **bài toán phân loại nhị phân (Binary Classification)**.

---

### 🔹 Mục tiêu
- Xây dựng mô hình dự đoán rủi ro tín dụng của khách hàng
- So sánh hiệu quả của các mô hình Machine Learning
- Đánh giá mô hình bằng các metric phù hợp với dữ liệu mất cân bằng
- Lựa chọn mô hình tối ưu cho bài toán

---

## 2. Dataset

### 🔹 Nguồn dữ liệu
- **Tên dataset:** German Credit Data
- **Nguồn:** Kaggle  
- **Link tải:**  
  https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk

---

### 🔹 Mô tả dataset
- **Số lượng mẫu:** ~1000
- **Số lượng thuộc tính:** 10 thuộc tính đầu vào
- **Biến mục tiêu (Target):** `Risk`
  - `Good` – khách hàng có rủi ro tín dụng thấp
  - `Bad` – khách hàng có rủi ro tín dụng cao

---

### 🔹 Nhóm thuộc tính
Các thuộc tính trong dataset phản ánh:
- Thông tin tài chính (số tiền vay, thời gian vay, thu nhập…)
- Hành vi và lịch sử tín dụng
- Thông tin cá nhân cơ bản

---

## 3. Pipeline xử lý dữ liệu

Pipeline của bài toán được xây dựng theo các bước sau:

[Input CSV] --> [Preprocess & Encode] --> [Train Models] --> [Save .pkl Files]
                                                                    |
                                                                    v
[User Input] --> [Load .pkl Files] --> [Transform Data] --> [Predict Result]

