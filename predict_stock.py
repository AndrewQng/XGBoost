import xgboost as xgb
import pandas as pd
import os

def main():
    print("=== PHẦN 2: DỰ ĐOÁN CHỨNG KHOÁN THỰC TẾ ===")
    
    model_path = 'stock_xgboost_model.json'
    new_data_file = 'stock_new_data.csv'
    
    if not os.path.exists(model_path):
        print(f"[LỖI] Chưa có bộ não '{model_path}'. Hãy chạy python train_stock.py trước.")
        return
        
    if not os.path.exists(new_data_file):
        print(f"[LỖI] Không tìm thấy dữ liệu mới '{new_data_file}'.")
        return
        
    # -------------------------------------------------------------
    # 1. TẢI LẠI MÔ HÌNH (LOAD MODEL) TỪ FILE JSON
    # -------------------------------------------------------------
    print(f"1. Đang lắp 'Bộ não' từ file '{model_path}'...")
    # Khởi tạo một mô hình trống
    model = xgb.XGBClassifier() 
    # Nạp kiến thức từ file json vào mô hình trống này
    model.load_model(model_path)
    
    # -------------------------------------------------------------
    # 2. ĐỌC DỮ LIỆU MỚI ĐỂ DỰ ĐOÁN
    # -------------------------------------------------------------
    print(f"2. Đang đọc bảng giá hôm nay từ '{new_data_file}'...")
    df_new = pd.read_csv(new_data_file)
    
    # Tách mã cổ phiếu ra một biến riêng để tí in ra màn hình cho dễ nhìn
    ma_co_phieu = df_new['MaCoPhieu']
    # Đầu vào cho mô hình AI phải giống hệt lúc học (Bỏ cột tên Mã đi)
    X_new = df_new.drop(columns=['MaCoPhieu'])
    
    # -------------------------------------------------------------
    # 3. YÊU CẦU MÔ HÌNH DỰ ĐOÁN
    # -------------------------------------------------------------
    # Dự đoán kết quả cuối cùng (0 hoặc 1)
    predictions = model.predict(X_new)
    # Lấy Xác suất % dự đoán (predict_proba trả về tỷ lệ [Giảm, Tăng])
    probabilities = model.predict_proba(X_new)[:, 1] # [:, 1] nghĩa là lấy cột số 2 (Xác suất Tăng)
    
    # Hiển thị kết quả cực kỳ đẹp mắt
    print("\n=> KẾT QUẢ DỰ ĐOÁN XU HƯỚNG NGÀY MAI:\n")
    print(f"{'MÃ CP':<8} | {'DỰ ĐOÁN':<12} | {'TỶ LỆ CHẮC CHẮN TĂNG'}")
    print("-" * 50)
    for i in range(len(ma_co_phieu)):
        if predictions[i] == 1:
            ket_qua = "📈 TĂNG LÊN"
        else:
            ket_qua = "📉 GIẢM XUỐNG"
            
        xac_suat = probabilities[i] * 100
        print(f"{ma_co_phieu[i]:<8} | {ket_qua:<12} | {xac_suat:.2f}%")

if __name__ == "__main__":
    main()
