import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def main():
    print("=== PHẦN 1: HUẤN LUYỆN MÔ HÌNH CHỨNG KHOÁN ===")
    data_file = 'stock_train_data.csv'
    
    if not os.path.exists(data_file):
        print(f"[LỖI] Không tìm thấy '{data_file}'. Hãy chạy python generate_stock_data.py trước.")
        return
        
    # 1. Đọc dữ liệu Lịch sử
    df = pd.read_csv(data_file)
    
    # 2. Định hình Input (X) và Output (y)
    y = df['TangGia']
    X = df.drop(columns=['TangGia'])
    
    # 3. Chia tập Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Khởi tạo XGBoost
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        objective='binary:logistic',
        random_state=42
    )
    
    # 5. Bắt đầu học
    print("Đang cho AI học từ dữ liệu lịch sử...")
    model.fit(X_train, y_train)
    
    # Đánh giá cơ bản
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"-> Độ chính xác của AI sau khi học: {acc*100:.2f}%")
    
    # -------------------------------------------------------------
    # 6. XUẤT MÔ HÌNH (SAVE MODEL) - QUAN TRỌNG NHẤT
    # -------------------------------------------------------------
    model_path = 'stock_xgboost_model.json'
    model.save_model(model_path)
    print(f"\n[THÀNH CÔNG] đã xuất và lưu model của AI tại file: '{model_path}'")

if __name__ == "__main__":
    main()
