# Đồ Án: Ứng dụng Học máy XGBoost Dự đoán Xu hướng Chứng Khoán

Dự án này là một nguyên mẫu (prototype) mô phỏng việc sử dụng thuật toán **XGBoost Classifier** trong Machine Learning để dự đoán xu hướng giá cổ phiếu (Tăng/Giảm) thông qua phân tích dữ liệu lịch sử và các chỉ báo kỹ thuật cơ bản.

## 1. Giới thiệu Kiến trúc & Công nghệ

- **Ngôn ngữ lập trình**: Python 3.10+
- **Thư viện cốt lõi**:
  - `xgboost`: Triển khai thuật toán Extreme Gradient Boosting mạnh mẽ cho bài toán phân lớp.
  - `pandas`: Thao tác và tiền xử lý dữ liệu dạng bảng.
  - `scikit-learn`: Đánh giá hiệu năng mô hình (Accuracy, Precision, Recall, F1) và chia tách dữ liệu.
- **Dữ liệu giả lập**: Bao gồm các đặc trưng (Features) như `GiaMoCua`, `KhoiLuong`, `BienDong`, `ChiSoRSI`.

## 2. Cấu trúc Thư mục

```text
.
├── data/                       # Thư mục lưu trữ dữ liệu
│   ├── stock_train_data.csv    # Dữ liệu lịch sử dùng để huấn luyện mô hình
│   └── stock_new_data.csv      # Dữ liệu mới dùng để kiểm tra dự đoán
├── models/                     # Thư mục chứa mô hình máy học đã xuất ra
│   └── stock_xgboost_model.json# Tệp "bộ não" lưu trọng số của AI
├── generate_stock_data.py      # Kịch bản sinh thêm dữ liệu giả lập (Mock data)
├── train_stock.py              # Kịch bản huấn luyện và đánh giá mô hình
├── predict_stock.py            # Kịch bản nạp mô hình và dự đoán dữ liệu thực
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md                   # Tài liệu hướng dẫn sử dụng
```

## 3. Hướng dẫn Cài đặt Môi trường

Khuyến khích tạo một môi trường ảo (Virtual Environment) trước khi chạy dự án. Sau đó cài đặt các thư viện thông qua pip:

```bash
pip install -r requirements.txt
```

## 4. Hướng dẫn Vận hành Hệ thống

### Bước 1: Khởi tạo và Sinh Dữ liệu (Tùy chọn)
Mặc định hệ thống đã có file mẫu. Tuy nhiên, để tạo bộ dữ liệu lớn hơn nhằm giúp mô hình học tập tốt hơn, bạn có thể chạy:
```bash
python generate_stock_data.py
```
*(Chương trình sẽ tự động sinh ngẫu nhiên 2000 dòng dữ liệu vào `data/stock_train_data.csv`)*.

### Bước 2: Huấn luyện Mô hình Học máy (Training)
Tiến hành cho AI học trên tập dữ liệu lịch sử bằng lệnh:
```bash
python train_stock.py
```
**Kết quả mong đợi:** 
- Chương trình tiến hành đọc dữ liệu, phân chia tập Train (80%) và Test (20%).
- Huấn luyện XGBoost và tự động in báo cáo hiệu suất chi tiết (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Mô hình lưu thành công vào file `models/stock_xgboost_model.json`.

### Bước 3: Dự báo Thực tế (Prediction)
Sau khi mô hình đã hội tụ và được lưu, hệ thống có thể dự báo các phiên tiếp theo dựa trên dữ liệu đầu vào mới ở `data/stock_new_data.csv`:
```bash
python predict_stock.py
```
**Kết quả mong đợi:** 
Bảng báo cáo trực quan dưới dạng Console, chỉ ra từng mã cổ phiếu sẽ Tăng/Giảm đi kèm mức độ tin cậy (Xác suất % phần trăm).

## 5. Tổng kết
Bài tập này trình bày một quy trình chuẩn hóa (Pipeline) cơ bản của Khoa học Dữ liệu, từ việc **Chuẩn bị Dữ liệu (Data Prep)** -> **Huấn luyện Mô hình (Model Training)** -> **Đánh giá (Evaluation)** -> **Triển khai Dự đoán (Inference)**. Bằng việc ứng dụng XGBoost - một thuật toán Boosting rất phổ biến hiện nay, dự án thể hiện cách máy tính học được các quy luật ẩn sâu bên trong các chỉ số tài chính.
