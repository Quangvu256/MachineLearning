import os
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight 
import xgboost as xgb
import json
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize
import imgaug as ia
from imgaug import augmenters as iaa
from scipy.stats import randint, uniform

# --- CẤU HÌNH ---

PROCESSED_DATA_DIR = 'ML'
MODEL_FILENAME = 'xgboost_rich_features_tuned.json'
LABEL_ENCODER_FILENAME = 'label_encoder_rich_features_tuned.json'
IMAGE_SIZE = (128, 128)



ia.seed(42)
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-15, 15),
    ),
    iaa.CoarseDropout((0.01, 0.05), size_percent=(0.02, 0.1)),
    iaa.Multiply((0.8, 1.2)),
    iaa.LinearContrast((0.8, 1.2)),
], random_order=True)


def extract_hog_features(pil_image):
    """Trích xuất đặc trưng hình dạng (HOG)"""
    gray_img = np.array(pil_image.convert('L'))
    gray_img_resized = resize(gray_img, IMAGE_SIZE)
    features = hog(gray_img_resized, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), visualize=False)
    return features

def extract_color_histogram(pil_image):
    """Trích xuất đặc trưng màu sắc"""
    img_resized = pil_image.resize(IMAGE_SIZE)
    hsv_img = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp_features(pil_image):
    """Trích xuất đặc trưng kết cấu (LBP)"""
    gray_img = np.array(pil_image.convert('L'))
    gray_img_resized = resize(gray_img, IMAGE_SIZE)
    
    # Cài đặt cho LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img_resized, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    # Chuẩn hóa histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# 1.3: Hàm tổng hợp để lấy tất cả đặc trưng
def get_combined_features(pil_image):
    """Gọi các hàm trích xuất và nối các vector đặc trưng lại"""
    hog_features = extract_hog_features(pil_image)
    color_features = extract_color_histogram(pil_image)
    lbp_features = extract_lbp_features(pil_image)
    
    # Nối tất cả lại thành một vector duy nhất
    combined = np.concatenate([hog_features, color_features, lbp_features])
    return combined

# 1.4: Hàm xử lý cho ảnh huấn luyện (Tăng cường + Trích xuất)
def process_training_image(image_path, label):
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        # Tạo 3 ảnh tăng cường
        augmented_images_np = augmenter(images=[img_np, img_np, img_np])
        
        # Xử lý ảnh gốc và 3 ảnh tăng cường
        all_images_to_process = [img] + [Image.fromarray(aug_np) for aug_np in augmented_images_np]
        feature_vectors = [get_combined_features(im) for im in all_images_to_process]
        labels = [label] * len(feature_vectors)
        return feature_vectors, labels
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh huấn luyện {image_path}: {e}")
        return [], []

# 1.5: Hàm xử lý cho ảnh kiểm tra (Chỉ trích xuất)
def process_test_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        return get_combined_features(img)
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh kiểm tra {image_path}: {e}")
        return None

# --- BƯỚC 2: TỐI ƯU SIÊU THAM SỐ & HUẤN LUYỆN MÔ HÌNH ---
# Cập nhật hàm để nhận thêm sample_weights
def tune_and_train_model(X_train, y_train, X_test, y_test, num_classes, sample_weights):
    print("\nBắt đầu quá trình tìm kiếm siêu tham số tối ưu...")
    
    param_dist = {
        'max_depth': randint(4, 12),
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5),
        'n_estimators': randint(200, 600)
    }

    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        device='cuda',
        tree_method='hist',
        eval_metric='merror',
        early_stopping_rounds=30 
    )

    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=30, 
        scoring='accuracy',
        n_jobs=1, 
        cv=3, 
        verbose=3,
        random_state=42
    )

    print("Đang chạy RandomizedSearchCV...")
    # Cập nhật lời gọi .fit() để truyền trọng số vào
    random_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], sample_weight=sample_weights, verbose=False)

    print("\nQuá trình tìm kiếm hoàn tất!")
    print("Siêu tham số tốt nhất được tìm thấy: ", random_search.best_params_)
    print(f"Độ chính xác tốt nhất trên tập validation (CV): {random_search.best_score_ * 100:.2f}%")

    best_model = random_search.best_estimator_

    print("\nĐánh giá mô hình tốt nhất trên tập kiểm tra...")
    preds = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"----------------------------------------------------")
    print(f"✅ Độ chính xác (Accuracy) cuối cùng trên tập test: {accuracy * 100:.2f}%")
    print(f"----------------------------------------------------")
    
    return best_model

# --- HÀM CHÍNH ĐỂ CHẠY TOÀN BỘ PIPELINE ---
def main():
    # 1. Quét dữ liệu
    image_paths, labels_str = [], []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    print(f"Đang quét dữ liệu từ thư mục '{PROCESSED_DATA_DIR}'...")

    if not os.path.isdir(PROCESSED_DATA_DIR):
        print(f"Lỗi: Thư mục '{PROCESSED_DATA_DIR}' không tồn tại.")
        return

    for class_name in sorted(os.listdir(PROCESSED_DATA_DIR)):
        class_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
        if os.path.isdir(class_dir):
            for item_name in os.listdir(class_dir):
                if item_name.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(class_dir, item_name))
                    labels_str.append(class_name)

    if not image_paths:
        print("Không tìm thấy file ảnh nào trong thư mục đã xử lý.")
        return
    print(f"Đã tìm thấy {len(image_paths)} file ảnh hợp lệ.")
    
    # 2. Mã hóa nhãn và chia tập dữ liệu
    print("Mã hóa nhãn và chia tập dữ liệu...")
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels_str)
    X_paths_train, X_paths_test, y_train, y_test = train_test_split(
        image_paths, labels_encoded, test_size=0.25, random_state=42, stratify=labels_encoded
    )

    # 3. Xử lý tập huấn luyện
    print(f"\nBắt đầu xử lý {len(X_paths_train)} ảnh huấn luyện (Tăng cường & Trích xuất đặc trưng phức hợp)...")
    train_results = Parallel(n_jobs=-1)(
        delayed(process_training_image)(path, label) 
        for path, label in tqdm(zip(X_paths_train, y_train), total=len(X_paths_train))
    )
    X_train_features, y_train_augmented = [], []
    for features, labels in train_results:
        X_train_features.extend(features)
        y_train_augmented.extend(labels)
    X_train = np.array(X_train_features)
    y_train = np.array(y_train_augmented)

    # 4. Xử lý tập kiểm tra
    print(f"\nBắt đầu xử lý {len(X_paths_test)} ảnh kiểm tra (Trích xuất đặc trưng phức hợp)...")
    test_results = Parallel(n_jobs=-1)(
        delayed(process_test_image)(path) 
        for path in tqdm(X_paths_test)
    )
    X_test_features = [f for f in test_results if f is not None]
    valid_indices = [i for i, f in enumerate(test_results) if f is not None]
    y_test = y_test[valid_indices]
    X_test = np.array(X_test_features)

    # 5. Tối ưu và Huấn luyện mô hình
    if len(X_train) > 0 and len(X_test) > 0:
        # *** THÊM BƯỚC TÍNH TOÁN TRỌNG SỐ ***
        print("\nTính toán trọng số mẫu để xử lý mất cân bằng dữ liệu...")
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        # Cập nhật lời gọi hàm để truyền trọng số vào
        model = tune_and_train_model(X_train, y_train, X_test, y_test, len(le.classes_), sample_weights)
        
        # 6. Lưu mô hình và bộ mã hóa
        print("Đang lưu mô hình và bộ mã hóa nhãn...")
        model.save_model(MODEL_FILENAME)
        classes_list = le.classes_.tolist()
        with open(LABEL_ENCODER_FILENAME, 'w') as f:
            json.dump(classes_list, f)
        print(f"Đã lưu mô hình vào file: '{MODEL_FILENAME}'")
        print(f"Đã lưu bộ mã hóa vào file: '{LABEL_ENCODER_FILENAME}'")
    else:
        print("Không có dữ liệu để huấn luyện.")

if __name__ == '__main__':
    main()
