import pandas as pd
import numpy as np
import os

def append_data():
    data_file = 'stock_train_data.csv'
    n_samples = 2000
    
    # Tạo dữ liệu ngẫu nhiên
    GiaMoCua = np.round(np.random.uniform(10.0, 100.0, n_samples), 1)
    KhoiLuong = np.round(np.random.uniform(10000.0, 50000.0, n_samples), 1)
    BienDong = np.round(np.random.uniform(0.0, 5.0, n_samples), 2)
    ChiSoRSI = np.round(np.random.uniform(20.0, 80.0, n_samples), 1)
    TangGia = np.random.randint(0, 2, n_samples)
    
    df_new = pd.DataFrame({
        'GiaMoCua': GiaMoCua,
        'KhoiLuong': KhoiLuong,
        'BienDong': BienDong,
        'ChiSoRSI': ChiSoRSI,
        'TangGia': TangGia
    })
    
    # Ghi thêm vào file CSV hiện có
    df_new.to_csv(data_file, mode='a', header=not os.path.exists(data_file), index=False)
    print(f"Đã thêm thành công {n_samples} dòng dữ liệu mẫu vào '{data_file}'")

if __name__ == '__main__':
    append_data()
