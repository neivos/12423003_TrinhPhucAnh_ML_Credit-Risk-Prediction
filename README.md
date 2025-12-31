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

Dataset → EDA → Clean → Encode → Train → Evaluate → Predict

## 4. Mô hình sử dụng

Các mô hình Machine Learning được sử dụng trong bài:

### 🔹 Logistic Regression (LR)
- Mô hình phân loại tuyến tính
- Được dùng làm mô hình tham chiếu (baseline)

### 🔹 Decision Tree (DT)
- Mô hình dựa trên cây quyết định
- Dễ hiểu, dễ giải thích
- Tuy nhiên dễ overfitting

### 🔹 Random Forest (RF)
- Mô hình ensemble kết hợp nhiều Decision Tree
- Giảm overfitting, độ ổn định cao
- Hoạt động tốt với dữ liệu dạng bảng

### 🔹 XGBoost (XGB)
- Thuật toán boosting mạnh
- Học tuần tự để giảm lỗi dự đoán
- Hiệu quả cao nhưng nhạy với tuning

  ## 5. Kết quả đánh giá

Do dataset có **mất cân bằng (Good ~70%, Bad ~30%)**, các metric được sử dụng bao gồm:

- Accuracy
- Precision
- Recall
- F1-score
- **ROC – AUC (metric chính)**

### 🔹 Nhận xét
- Random Forest đạt **ROC–AUC cao nhất**
- XGBoost cho kết quả tốt nhưng chưa vượt Random Forest
- Decision Tree dùng làm baseline để so sánh

---

