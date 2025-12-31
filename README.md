# DỰ ĐOÁN RỦI RO TÍN DỤNG KHÁCH HÀNG - MACHINE LEARNING

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
## 6. Hướng dẫn chạy chương trình

Để chạy dự án, vui lòng thực hiện tuần tự các bước sau:

### 1. Cài đặt môi trường
Khuyến khích sử dụng môi trường ảo (Virtual Environment) để tránh xung đột thư viện.

**Bước 1.1: Clone dự án về máy**
```bash
git clone <link-repo>
cd Credit-Risk-Prediction
```
**Bước 1.2: Cài đặt thư viện**
```bash
pip install -r requirements.txt
```
### 2. Huấn luyện mô hình
Trước khi chạy Demo, bạn cần chạy file train để xử lý dữ liệu và tạo file model
```bash
python train_model.py
```
Màn hình thông báo thành công. Thư mục data/pkl/ sẽ xuất hiện chứa các file model.
### 3. Chạy demo
Khởi động ứng dụng web Streamlit.
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở tại: http://localhost:8501.
Nhập thông tin và bấm "Dự báo" để xem kết quả.

## Tác giả
Họ tên: Trịnh Phúc Anh
Mã Sinh viên  : 12423003
Lớp : 12423TN
Trường Đại học sư phạm kỹ thuật Hưng Yên
####
Dự án được thực hiện với mục đích học tập và nghiên cứu Machine Learning, không sử dụng cho mục đích thương mại.
