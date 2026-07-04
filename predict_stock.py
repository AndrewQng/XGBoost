import os
import sys
import io
import glob
import pandas as pd
import xgboost as xgb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def predict_stock_trend() -> None:
    """
    Tải mô hình XGBoost đã được huấn luyện, đọc dữ liệu thị trường mới,
    và thực hiện dự đoán xu hướng cổ phiếu (Tăng/Giảm) kèm theo xác suất.
    """
    print("==================================================")
    print("   PHẦN 2: DỰ ĐOÁN CHỨNG KHOÁN THỰC TẾ (PREDICT)")
    print("==================================================\n")
    
    model_dir = 'models'
    model_files = glob.glob(os.path.join(model_dir, 'stock_xgboost_model*.json'))
    
    # Kiểm tra tính toàn vẹn của file
    if not model_files:
        print(f"[LỖI] Không tìm thấy mô hình đã huấn luyện trong thư mục '{model_dir}'.")
        print("Vui lòng chạy 'python train_stock.py' trước để huấn luyện và lưu mô hình.")
        return
        
    # Lấy mô hình mới nhất dựa trên thời gian tạo
    model_path = max(model_files, key=os.path.getctime)
    new_data_file = os.path.join('data', 'stock_new_data.csv')
        
    if not os.path.exists(new_data_file):
        print(f"[LỖI] Không tìm thấy dữ liệu cần dự đoán tại '{new_data_file}'.")
        return
        
    # -------------------------------------------------------------
    # 1. TẢI MÔ HÌNH TỪ ĐĨA CỨNG (LOAD MODEL)
    # -------------------------------------------------------------
    print(f"[1/3] Đang tải mô hình học máy từ '{model_path}'...")
    model = xgb.XGBClassifier() 
    model.load_model(model_path)
    
    # -------------------------------------------------------------
    # 2. ĐỌC DỮ LIỆU ĐẦU VÀO MỚI (INPUT DATA)
    # -------------------------------------------------------------
    print(f"[2/3] Đang nạp dữ liệu phiên giao dịch mới từ '{new_data_file}'...")
    df_new = pd.read_csv(new_data_file)
    
    # Giữ lại danh sách Mã Cổ Phiếu để hiển thị trong kết quả
    ma_co_phieu = df_new['MaCoPhieu']
    
    # Loại bỏ cột định danh (Mã) để định hình lại dữ liệu khớp với lúc Training
    X_new = df_new.drop(columns=['MaCoPhieu'])
    
    # -------------------------------------------------------------
    # 3. CHẠY DỰ ĐOÁN BẰNG THUẬT TOÁN XGBOOST
    # -------------------------------------------------------------
    print("[3/3] Đang xử lý tính toán và phân loại...\n")
    
    # Lấy mảng nhãn dự đoán tuyệt đối (0 hoặc 1)
    predictions = model.predict(X_new)
    # Lấy mảng xác suất dự đoán (chỉ lấy cột index 1 đại diện cho nhãn 1 = Tăng)
    probabilities = model.predict_proba(X_new)[:, 1]
    
    # -------------------------------------------------------------
    # 4. IN KẾT QUẢ ĐẦU RA (OUTPUT BÁO CÁO) VÀ LƯU FILE TXT
    # -------------------------------------------------------------
    print("=> BÁO CÁO KẾT QUẢ DỰ ĐOÁN XU HƯỚNG TƯƠNG LAI:\n")
    header = f"{'MÃ CP':<10} | {'DỰ ĐOÁN XU HƯỚNG':<18} | {'ĐỘ TIN CẬY (XÁC SUẤT TĂNG)'}"
    separator = "-" * 65
    
    print(header)
    print(separator)
    
    model_basename = os.path.basename(model_path).replace('.json', '')
    result_file = f"predict_result_{model_basename}.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("BÁO CÁO KẾT QUẢ DỰ ĐOÁN XU HƯỚNG CỔ PHIẾU\n")
        f.write("=========================================\n\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        
        for i in range(len(ma_co_phieu)):
            if predictions[i] == 1:
                ket_qua = "📈 TĂNG"
            else:
                ket_qua = "📉 GIẢM"
                
            xac_suat = probabilities[i] * 100
            line = f"{ma_co_phieu[i]:<10} | {ket_qua:<18} | {xac_suat:.2f}%"
            print(line)
            f.write(line + "\n")
            
    print(f"\n[THÀNH CÔNG] Đã lưu báo cáo kết quả vào file: '{result_file}'")
    print("\n==================================================")
    print("              HOÀN THÀNH DỰ ĐOÁN                  ")
    print("==================================================")

if __name__ == "__main__":
    predict_stock_trend()
