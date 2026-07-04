# Dự đoán Xu Hướng Chứng Khoán Bằng XGBoost

Đây là dự án ví dụ sử dụng thư viện **XGBoost** để xây dựng mô hình AI dự đoán xu hướng tăng/giảm của chứng khoán dựa trên dữ liệu lịch sử.

## Cài đặt Thư Viện

Hãy chắc chắn rằng bạn đã cài đặt các thư viện cần thiết. Chạy lệnh sau để cài đặt trong Terminal:

```bash
pip install xgboost pandas scikit-learn
```

## Hướng dẫn chạy (How to run)

Dự án này gồm 2 phần chính: Huấn luyện mô hình (AI Học) và Dự đoán thực tế.

### Bước 1: Huấn luyện mô hình (Train Model)

Đầu tiên, AI cần học từ dữ liệu lịch sử (`stock_train_data.csv`).
Hãy chạy lệnh sau trên Terminal:

```bash
python train_stock.py
```

*Kết quả:* Mô hình sẽ tự học, in ra độ chính xác và tự động lưu "bộ não" của nó vào một file cấu hình có tên là `stock_xgboost_model.json`.

### Bước 2: Dự đoán xu hướng (Predict)

Sau khi AI đã được huấn luyện xong ở Bước 1, bạn có thể dùng nó để dự đoán xu hướng cổ phiếu hôm nay bằng cách cung cấp dữ liệu mới trong file `stock_new_data.csv`.
Tiếp tục chạy lệnh sau:

```bash
python predict_stock.py
```

*Kết quả:* Chương trình sẽ in ra bảng dự đoán chi tiết xem mã cổ phiếu nào sẽ tăng, mã nào sẽ giảm kèm theo tỷ lệ % chắc chắn.

---
**Lưu ý**: Dữ liệu trong project này (`stock_train_data.csv` và `stock_new_data.csv`) là dữ liệu được tạo ngẫu nhiên (mock data) nhằm mục đích demo cách dùng thư viện XGBoost cơ bản nhất.
