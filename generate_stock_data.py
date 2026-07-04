import pandas as pd
import numpy as np
import os
import sys

def append_data(n_samples: int = 2000) -> None:
    """
    Sinh ngẫu nhiên dữ liệu chứng khoán giả lập và ghi thêm vào file CSV.

    Args:
        n_samples (int): Số lượng mẫu (dòng dữ liệu) cần sinh thêm. Mặc định là 2000.

    Returns:
        None
    """
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    data_file = os.path.join(data_dir, 'stock_train_data.csv')
    
    # Thiết lập seed để có thể tái lập kết quả nếu cần (tuỳ chọn)
    # np.random.seed(42)
    
    # Khởi tạo các đặc trưng giả lập
    gia_mo_cua = np.round(np.random.uniform(10.0, 100.0, n_samples), 1)
    khoi_luong = np.round(np.random.uniform(10000.0, 50000.0, n_samples), 1)
    bien_dong = np.round(np.random.uniform(0.0, 5.0, n_samples), 2)
    chi_so_rsi = np.round(np.random.uniform(20.0, 80.0, n_samples), 1)
    
    # Khởi tạo nhãn (0: Giảm, 1: Tăng)
    tang_gia = np.random.randint(0, 2, n_samples)
    
    # Tạo DataFrame từ thư viện Pandas
    df_new = pd.DataFrame({
        'GiaMoCua': gia_mo_cua,
        'KhoiLuong': khoi_luong,
        'BienDong': bien_dong,
        'ChiSoRSI': chi_so_rsi,
        'TangGia': tang_gia
    })
    
    # Ghi dữ liệu vào cuối file CSV (mode 'a' = append)
    # Nếu file chưa tồn tại, ghi cả Header
    file_exists = os.path.exists(data_file)
    df_new.to_csv(data_file, mode='a', header=not file_exists, index=False)
    
    # Xử lý an toàn cho Terminal không hỗ trợ UTF-8 (ví dụ trên Windows)
    message = f"Đã thêm thành công {n_samples} dòng dữ liệu mẫu vào '{data_file}'"
    try:
        print(message)
    except UnicodeEncodeError:
        # Nếu lỗi encode, in ra bản không dấu
        print(f"Da them thanh cong {n_samples} dong du lieu mau vao '{data_file}'")

if __name__ == '__main__':
    print("=== SCRIPT SINH DỮ LIỆU CHỨNG KHOÁN ===")
    append_data(2000)
