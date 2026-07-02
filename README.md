# Phân loại 5 Lớp Chim bằng Đặc trưng Cổ điển và XGBoost

Dự án này là một pipeline hoàn chỉnh để huấn luyện một mô hình học máy nhằm phân loại 5 loài chim từ bộ dữ liệu CUB-200-2011.
Pipeline sử dụng các kỹ thuật trích xuất đặc trưng cổ điển (HOG, LBP, HSV), tăng cường dữ liệu và tối ưu siêu tham số với XGBoost.

## 📂 Cấu trúc Dự án

```
MachineLearning/
├── README.md                    # Tài liệu hướng dẫn
├── requirements.txt             # Các thư viện cần thiết
├── config.py                    # Cấu hình chung và tham số (auto-detect CPU/CUDA)
├── features.py                  # Module trích xuất đặc trưng dùng chung (HOG, LBP, HSV)
├── preprocessing.py             # Tiền xử lý dữ liệu (tải dataset, cắt Bounding Box)
├── train.py                     # Script huấn luyện mô hình XGBoost
├── inference.py                 # Pipeline dự đoán cho ứng dụng
├── app.py                       # Ứng dụng Streamlit
├── models/                      # Thư mục chứa các model artifacts (sẽ được sinh ra sau khi train)
│   ├── xgboost_model.json
│   ├── label_encoder.json
│   └── feature_scaler.pkl
├── ML/                          # Thư mục chứa ảnh đã tiền xử lý
└── tests/                       # Unit tests
    └── test_features.py
```

## ⚙️ Cài đặt

Cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

## 🚀 Hướng dẫn Chạy

### Bước 1: Tiền xử lý và cắt ảnh theo Bounding Box

```bash
python preprocessing.py
```
*Lưu ý: Mặc định script sẽ tải CUB-200-2011. Bạn có thể cấu hình thư mục lưu dataset qua biến môi trường `CUB_DATA_DIR`.*

Sau khi chạy xong, copy 5 lớp chim (hoặc các lớp bạn muốn) vào thư mục `ML/`.
Hiện tại `ML/` đã có sẵn dữ liệu của 5 lớp.

### Bước 2: Huấn luyện mô hình

```bash
python train.py
```
Script sẽ tự động:
1. Phát hiện thiết bị chạy (CPU/CUDA).
2. Trích xuất đặc trưng và tăng cường dữ liệu.
3. Chuẩn hóa đặc trưng với `StandardScaler`.
4. Tìm siêu tham số tốt nhất bằng `RandomizedSearchCV`.
5. Lưu model artifacts vào thư mục `models/`.

### Bước 3: Chạy ứng dụng Demo

```bash
streamlit run app.py
```

## 🧪 Testing

Chạy unit tests để đảm bảo tính ổn định của pipeline đặc trưng:

```bash
pytest tests/ -v
```

## 🛠️ Chi tiết Kỹ thuật

- **Đặc trưng hợp nhất**:
  - HOG (Hình dạng): pixels_per_cell=(16,16), cells_per_block=(2,2).
  - LBP (Kết cấu): radius=3, n_points=24, method="uniform".
  - HSV Histogram (Màu sắc): 8x8x8 bins.
- **Tiền xử lý & Huấn luyện**:
  - Resize ảnh bằng PIL để nhất quán giữa train và inference.
  - Sử dụng `StandardScaler`.
  - Fix Data Leakage bằng cách tách biệt `RandomizedSearchCV` và quá trình Early Stopping.
## Link Demo:
