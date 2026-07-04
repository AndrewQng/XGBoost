import os
import sys
import io
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train_xgboost_model() -> None:
    """
    Đọc dữ liệu lịch sử từ file CSV, tiền xử lý, huấn luyện mô hình phân loại XGBoost
    và đánh giá hiệu năng mô hình (Accuracy, Precision, Recall, F1, Confusion Matrix).
    Cuối cùng, lưu mô hình đã huấn luyện vào thư mục 'models'.
    """
    print("==================================================")
    print("   PHẦN 1: HUẤN LUYỆN MÔ HÌNH CHỨNG KHOÁN (TRAIN)")
    print("==================================================\n")
    
    data_file = os.path.join('data', 'stock_train_data.csv')
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'stock_xgboost_model.json')
    
    if not os.path.exists(data_file):
        print(f"[LỖI] Không tìm thấy dữ liệu tại: '{data_file}'")
        print("Vui lòng chạy 'python generate_stock_data.py' trước để sinh dữ liệu mẫu.")
        return
        
    # Tạo thư mục chứa mô hình nếu chưa có
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # 1. Đọc dữ liệu Lịch sử
    print(f"[1/5] Đang đọc dữ liệu từ '{data_file}'...")
    df = pd.read_csv(data_file)
    print(f"      Tổng số mẫu dữ liệu: {len(df)} dòng.")
    
    # 2. Định hình Đặc trưng (X) và Nhãn dự đoán (y)
    print("[2/5] Đang tách đặc trưng (X) và nhãn dự đoán (y)...")
    y = df['TangGia']
    X = df.drop(columns=['TangGia'])
    
    # 3. Phân chia tập Huấn luyện (Train) và tập Kiểm thử (Test)
    # 80% để học, 20% để đánh giá
    print("[3/5] Đang phân chia tập Train (80%) và Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Khởi tạo mô hình XGBoost
    print("[4/5] Đang khởi tạo thuật toán XGBoost và bắt đầu huấn luyện...")
    model = xgb.XGBClassifier(
        n_estimators=150,           # Số lượng cây quyết định
        max_depth=4,                # Độ sâu tối đa của mỗi cây
        learning_rate=0.05,         # Tốc độ học
        objective='binary:logistic',# Bài toán phân loại nhị phân (Tăng/Giảm)
        random_state=42,            # Cố định random_state để có thể tái lập kết quả
        eval_metric='logloss'       # Metric đánh giá tối ưu trong quá trình học
    )
    
    # Bắt đầu học (Fitting)
    model.fit(X_train, y_train)
    
    # Đánh giá hiệu năng trên tập Test
    print("\n[ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ]")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f" -> Độ chính xác tổng thể (Accuracy) : {acc * 100:.2f}%")
    print(f" -> Độ chuẩn xác (Precision)         : {prec * 100:.2f}%")
    print(f" -> Độ bao phủ (Recall)              : {rec * 100:.2f}%")
    print(f" -> Điểm F1-Score                    : {f1 * 100:.2f}%")
    print(" -> Ma trận nhầm lẫn (Confusion Matrix):")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")
    
    # 5. Lưu (Xuất) mô hình
    print("\n[5/5] Đang xuất mô hình ra tệp JSON...")
    model.save_model(model_path)
    print(f"[THÀNH CÔNG] Đã lưu trữ 'bộ não' của AI tại: '{model_path}'\n")

if __name__ == "__main__":
    train_xgboost_model()
