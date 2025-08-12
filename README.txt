Phân loại 5 Lớp Chim bằng Đặc trưng Cổ điển và XGBoost
Dự án này là một pipeline hoàn chỉnh để huấn luyện một mô hình học máy nhằm phân loại 5 loài chim từ bộ dữ liệu CUB-200-2011. Pipeline sử dụng các kỹ thuật trích xuất đặc trưng cổ điển (HOG, LBP, HSV), tăng cường dữ liệu và tối ưu siêu tham số với XGBoost.

🎯 Mục tiêu
Phân loại chính xác 5 lớp chim được chọn từ bộ dữ liệu CUB-200-2011, sử dụng pipeline xử lý ảnh và mô hình Gradient Boosting.

📂 Cấu trúc & Vai trò các File
Dự án bao gồm 2 script chính:

preprocessing.py:

Vai trò: Chuẩn bị dữ liệu.

Chức năng: Tự động đọc metadata từ bộ dữ liệu CUB-200-2011, cắt (crop) chính xác từng con chim ra khỏi ảnh nền dựa trên tọa độ hộp giới hạn (bounding box), và lưu các ảnh đã xử lý vào một thư mục mới (CUB_200_processed) với cấu trúc phân loại theo từng lớp.

train_xgboost.py:

Vai trò: Huấn luyện và đánh giá mô hình.

Chức năng: Quét qua các ảnh đã được tiền xử lý, áp dụng các kỹ thuật tăng cường dữ liệu để làm giàu tập huấn luyện, trích xuất một vector đặc trưng phức hợp (HOG + LBP + HSV), sau đó sử dụng RandomizedSearchCV để tự động tìm ra bộ tham số tốt nhất và huấn luyện mô hình XGBoost. Cuối cùng, script sẽ in ra độ chính xác và lưu lại mô hình đã huấn luyện.

⚙️ Cài đặt
Mở terminal hoặc command prompt và chạy lệnh sau để cài đặt tất cả các thư viện cần thiết:

pip install numpy pandas pillow joblib tqdm scikit-learn xgboost opencv-python scikit-image imgaug scipy

🚀 Hướng dẫn Chạy
Bước 1: Tiền xử lý và Cắt ảnh theo Bounding Box
Chạy script tiền xử lý để chuẩn bị dữ liệu. Script này sẽ đọc từ bộ dữ liệu gốc và tạo ra một thư mục mới chứa các ảnh đã được cắt.

python preprocessing.py

Kết quả: Một thư mục mới có tên CUB_200_processed sẽ được tạo, chứa 200 thư mục con tương ứng với 200 loài chim.

Bước 2: Chọn 5 Lớp để Huấn luyện
Bạn có thể chọn 5 lớp để huấn luyện bằng một trong hai cách sau:

Cách 1 (Nhanh - Chỉnh sửa code): Mở file preprocessing.py, tìm đến hàm main() và thêm đoạn code sau ngay sau khi đọc dữ liệu để lọc DataFrame:

# Danh sách các lớp bạn muốn huấn luyện
SELECTED_CLASSES = [
    "001.Black_footed_Albatross",
    "002.Laysan_Albatross",
    "003.Sooty_Albatross",
    "004.Groove_billed_Ani",
    "005.Crested_Auklet",
]

print(f"Đang lọc để chỉ giữ lại {len(SELECTED_CLASSES)} lớp đã chọn...")
data = data[data['class_name'].isin(SELECTED_CLASSES)].copy()

Cách 2 (Thủ công): Sau khi chạy xong preprocessing.py, tạo một thư mục mới (ví dụ: CUB_200_processed_5) và chỉ sao chép 5 thư mục con của 5 loài chim bạn muốn vào đó.

Bước 3: Huấn luyện Mô hình
Mở file train_xgboost.py và cập nhật biến PROCESSED_DATA_DIR để trỏ đến thư mục dữ liệu của bạn.

# Trỏ đến thư mục chứa 200 lớp hoặc 5 lớp đã chọn
PROCESSED_DATA_DIR = "CUB_200_processed" 
# hoặc PROCESSED_DATA_DIR = "CUB_200_processed_5"

Chạy script huấn luyện:

python train_xgboost.py

Kết quả: Script sẽ thực hiện quá trình tìm kiếm siêu tham số, huấn luyện, và cuối cùng in ra độ chính xác trên tập kiểm tra. Hai file kết quả sẽ được lưu lại:

xgboost_rich_features_tuned.json: Mô hình đã huấn luyện.

label_encoder_rich_features_tuned.json: Bộ mã hóa nhãn.

🛠️ Chi tiết Kỹ thuật
Tiền xử lý
Metadata: Đọc và gộp thông tin từ images.txt, image_class_labels.txt, classes.txt, và bounding_boxes.txt.

Xử lý Bounding Box: Cắt ảnh theo tọa độ (x, y, width, height).

Tăng tốc: Sử dụng joblib.Parallel(n_jobs=-1) để tận dụng tất cả các nhân CPU, tăng tốc đáng kể quá trình xử lý.

Đặc trưng & Tăng cường Dữ liệu
Kích thước chuẩn hóa: Mọi ảnh đều được resize về 128x128 trước khi trích xuất đặc trưng.

Đặc trưng phức hợp: Mỗi ảnh được biểu diễn bằng một vector duy nhất là sự kết hợp của:

HOG (Hình dạng): pixels_per_cell=(16,16), cells_per_block=(2,2).

LBP (Kết cấu): radius=3, n_points=24, phương pháp uniform.

HSV Histogram (Màu sắc): bins=[8, 8, 8].

Tăng cường dữ liệu: Mỗi ảnh huấn luyện sẽ tạo ra 3 biến thể mới bằng các kỹ thuật: lật, biến đổi affine (co giãn, dịch chuyển, xoay), xóa mảng (coarse dropout), thay đổi độ sáng và độ tương phản.

Huấn luyện
Phân chia dữ liệu: Chia tập dữ liệu thành train (75%) và test (25%) theo phương pháp stratify để đảm bảo tỷ lệ các lớp được giữ nguyên.

Mô hình: XGBoost với objective='multi:softmax', tree_method='hist', và early_stopping_rounds=30.

Tối ưu siêu tham số: RandomizedSearchCV được dùng để thử nghiệm 30 bộ tham số ngẫu nhiên trong không gian sau:

max_depth: [4, 12]

learning_rate: [0.01, 0.21)

subsample, colsample_bytree: [0.6, 1.0)

gamma: [0, 0.5)

n_estimators: [200, 600]

⚠️ Lưu ý
GPU (CUDA): Script huấn luyện được cấu hình mặc định để chạy trên GPU (device='cuda'). Nếu máy của bạn không có card đồ họa NVIDIA hỗ trợ CUDA, hãy mở file train_xgboost.py và xóa dòng device='cuda' trong phần khởi tạo XGBClassifier. Mô hình sẽ tự động chạy trên CPU.