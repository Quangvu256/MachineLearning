Phân loại 5 Lớp Chim bằng Đặc trưng Cổ điển và XGBoost

Dự án này là một pipeline hoàn chỉnh để huấn luyện một mô hình học máy nhằm phân loại 5 loài chim từ bộ dữ liệu CUB-200-2011. Pipeline sử dụng các kỹ thuật trích xuất đặc trưng cổ điển (HOG, LBP, HSV), tăng cường dữ liệu và tối ưu siêu tham số với XGBoost. Trong thực nghiệm, mô hình đạt hiệu quả phân lớp 5 lớp chim ở mức 88–90% trong 10 lần thử.

🎯 Mục tiêu

Phân loại chính xác 5 lớp chim được chọn từ bộ dữ liệu CUB-200-2011, sử dụng pipeline xử lý ảnh và mô hình Gradient Boosting.

📂 Cấu trúc & Vai trò các File

Dự án bao gồm 2 script chính:

preprocessing.py

Vai trò: Chuẩn bị dữ liệu.

Chức năng: Tự động đọc metadata từ CUB-200-2011, cắt (crop) chính xác từng con chim khỏi nền dựa trên bounding box, và lưu ảnh đã xử lý vào thư mục mới CUB_200_processed theo từng lớp.

train_xgboost.py

Vai trò: Huấn luyện và đánh giá mô hình.

Chức năng: Quét ảnh đã tiền xử lý, áp dụng tăng cường dữ liệu, trích xuất vector đặc trưng hợp nhất (HOG + LBP + HSV), dùng RandomizedSearchCV để tìm siêu tham số tốt nhất, huấn luyện XGBoost, in độ chính xác và lưu mô hình.

⚙️ Cài đặt
pip install numpy pandas pillow joblib tqdm scikit-learn xgboost opencv-python scikit-image imgaug scipy

🚀 Hướng dẫn Chạy

Bước 1: Tiền xử lý và cắt ảnh theo Bounding Box

python preprocessing.py


Kết quả: Tạo thư mục CUB_200_processed chứa 200 thư mục con tương ứng 200 loài chim.

Bước 2: Chọn 5 lớp để huấn luyện

Cách 1 (Nhanh – Chỉnh code trong preprocessing.py):

SELECTED_CLASSES = [
    "001.Black_footed_Albatross",
    "002.Laysan_Albatross",
    "003.Sooty_Albatross",
    "004.Groove_billed_Ani",
    "005.Crested_Auklet",
]
print(f"Đang lọc để chỉ giữ lại {len(SELECTED_CLASSES)} lớp đã chọn...")
data = data[data['class_name'].isin(SELECTED_CLASSES)].copy()


Cách 2 (Thủ công): Sau khi chạy xong preprocessing.py, tạo thư mục mới (ví dụ CUB_200_processed_5) và chỉ sao chép 5 thư mục con mong muốn vào đó.

Bước 3: Huấn luyện mô hình

Cập nhật đường dẫn dữ liệu trong train_xgboost.py:

PROCESSED_DATA_DIR = "CUB_200_processed"
# hoặc: PROCESSED_DATA_DIR = "CUB_200_processed_5"


Chạy huấn luyện:

python train_xgboost.py


Kết quả: Script sẽ tìm siêu tham số, huấn luyện và in độ chính xác trên tập kiểm tra. Lưu 2 file:

xgboost_rich_features_tuned.json (mô hình)

label_encoder_rich_features_tuned.json (mã hoá nhãn)

📈 Kết quả thực nghiệm: Mô hình đạt hiệu quả phân lớp 5 lớp chim ở mức 88–90% trong 10 lần thử.

🛠️ Chi tiết Kỹ thuật

Tiền xử lý

Metadata: đọc/gộp images.txt, image_class_labels.txt, classes.txt, bounding_boxes.txt.

Bounding Box: cắt ảnh theo (x, y, width, height).

Tăng tốc: joblib.Parallel(n_jobs=-1) tận dụng đa lõi CPU.

Đặc trưng & Tăng cường dữ liệu

Resize: mọi ảnh về 128×128.

Vector đặc trưng hợp nhất:

HOG (Hình dạng): pixels_per_cell=(16,16), cells_per_block=(2,2).

LBP (Kết cấu): radius=3, n_points=24, method="uniform".

HSV Histogram (Màu sắc): bins=[8, 8, 8].

Tăng cường dữ liệu: mỗi ảnh train sinh 3 biến thể (flip, affine, coarse dropout, thay đổi sáng/contrast).

Huấn luyện

Chia dữ liệu: train=75%, test=25% với stratify.

Mô hình: XGBoost objective='multi:softmax', tree_method='hist', early_stopping_rounds=30.

Tối ưu siêu tham số (30 cấu hình ngẫu nhiên):

max_depth: [4, 12]

learning_rate: [0.01, 0.21)

subsample, colsample_bytree: [0.6, 1.0)

gamma: [0, 0.5)

n_estimators: [200, 600]

⚠️ Lưu ý về GPU

Script mặc định dùng GPU (device='cuda'). Nếu không có CUDA, hãy xoá cấu hình này trong XGBClassifier để chạy CPU.
