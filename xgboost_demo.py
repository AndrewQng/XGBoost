import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import sys

def main():
    print("--- XGBOOST: BÀI TOÁN DỰ ĐOÁN HỌC SINH ĐỖ/TRƯỢT ---")
    
    # 1. Khai báo file dữ liệu
    data_file = 'du_lieu_hoc_sinh.csv'
    
    # Kiểm tra xem file CSV đã tồn tại chưa
    if not os.path.exists(data_file):
        print(f"[LỖI] Không tìm thấy file dữ liệu '{data_file}'.")
        print("Vui lòng chạy lệnh: 'python generate_data.py' trước để tạo file dữ liệu!")
        sys.exit(1)

    # 2. Đọc dữ liệu từ file CSV
    print(f"\n1. Đang đọc dữ liệu từ '{data_file}'...")
    df = pd.read_csv(data_file)
    
    # Tách Input (X) và Output (y)
    # y là cột 'KetQua', X là các cột còn lại
    y = df['KetQua']
    X = df.drop(columns=['KetQua'])
    
    # 3. Chia dữ liệu để Train/Test (80% học, 20% thi)
    print("2. Đang chia tập dữ liệu...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Khởi tạo mô hình
    print("3. Khởi tạo mô hình XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1
    )
    
    # 5. Huấn luyện (Training)
    print("4. Đang huấn luyện mô hình (Training)...")
    model.fit(X_train, y_train)
    
    # 6. Dự đoán & Đánh giá trên tập Test
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=> KẾT QUẢ ĐÁNH GIÁ: Mô hình dự đoán chuẩn xác: {accuracy * 100:.2f}%\n")
    
    # -------------------------------------------------------------
    # 7. DỰ ĐOÁN THỰC TẾ
    # -------------------------------------------------------------
    print("--- DỰ ĐOÁN HỌC SINH MỚI ---")
    hoc_sinh_moi = pd.DataFrame({
        'GioHoc': [8.0],        # Học 8 tiếng/ngày
        'GioChoiGame': [1.0],   # Chơi game 1 tiếng/ngày
        'DiemThiThu': [7.5]     # Điểm thi thử đạt 7.5
    })
    
    du_doan = model.predict(hoc_sinh_moi)
    xac_suat = model.predict_proba(hoc_sinh_moi)[0][1] # Lấy % tỷ lệ Đỗ
    
    ket_qua_chu = "ĐỖ" if du_doan[0] == 1 else "TRƯỢT"
    print(f"Thông tin: Học 8h/ngày, Chơi game 1h/ngày, Điểm thi thử 7.5")
    print(f"-> Máy tính dự đoán: {ket_qua_chu} (Với xác suất đỗ là: {xac_suat * 100:.2f}%)")

if __name__ == "__main__":
    main()
