import os
import random
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight 
import xgboost as xgb
import json
import joblib
import imgaug as ia
from imgaug import augmenters as iaa
from scipy.stats import randint, uniform

from config import (
    DEVICE, SEED, TEST_SIZE, CV_FOLDS, N_SEARCH_ITER, EARLY_STOPPING_ROUNDS,
    MODEL_PATH, ENCODER_PATH, SCALER_PATH, PROCESSED_DATA_DIR, MODEL_DIR
)
from features import get_combined_features

# Đặt seed cho reproducibility
random.seed(SEED)
np.random.seed(SEED)
ia.seed(SEED)

print(f"🖥️ Device: {DEVICE}")

# Tạo thư mục models nếu chưa có
os.makedirs(MODEL_DIR, exist_ok=True)

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

def process_test_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        return get_combined_features(img)
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh kiểm tra {image_path}: {e}")
        return None

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

    # ===== BƯỚC 1: Tìm hyperparams (KHÔNG dùng early_stopping/eval_set) =====
    xgb_search = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        device=DEVICE,
        tree_method='hist',
        eval_metric='merror',
        random_state=SEED
    )

    random_search = RandomizedSearchCV(
        xgb_search,
        param_distributions=param_dist,
        n_iter=N_SEARCH_ITER,
        scoring='accuracy',
        n_jobs=1,
        cv=CV_FOLDS,
        verbose=3,
        random_state=SEED
    )

    print("Đang chạy RandomizedSearchCV...")
    random_search.fit(X_train, y_train, sample_weight=sample_weights)

    print("\nQuá trình tìm kiếm hoàn tất!")
    print("Siêu tham số tốt nhất được tìm thấy: ", random_search.best_params_)
    print(f"Độ chính xác tốt nhất trên tập validation (CV): {random_search.best_score_ * 100:.2f}%")

    # ===== BƯỚC 2: Train final model với best params + early stopping =====
    best_params = random_search.best_params_
    
    final_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        device=DEVICE,
        tree_method='hist',
        eval_metric='merror',
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=SEED,
        **best_params
    )

    print("\nĐang huấn luyện mô hình cuối cùng...")
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        sample_weight=sample_weights,
        verbose=False
    )

    print("\nĐánh giá mô hình tốt nhất trên tập kiểm tra...")
    preds = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"----------------------------------------------------")
    print(f"✅ Độ chính xác (Accuracy) cuối cùng trên tập test: {accuracy * 100:.2f}%")
    print(f"----------------------------------------------------")
    
    return final_model

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
        image_paths, labels_encoded, test_size=TEST_SIZE, random_state=SEED, stratify=labels_encoded
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

    # Thêm StandardScaler
    print("\nÁp dụng StandardScaler cho các đặc trưng...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Lưu scaler
    print("Đang lưu StandardScaler...")
    joblib.dump(scaler, SCALER_PATH)

    # 5. Tối ưu và Huấn luyện mô hình
    if len(X_train) > 0 and len(X_test) > 0:
        print("\nTính toán trọng số mẫu để xử lý mất cân bằng dữ liệu...")
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        model = tune_and_train_model(X_train, y_train, X_test, y_test, len(le.classes_), sample_weights)
        
        # 6. Lưu mô hình và bộ mã hóa
        print("Đang lưu mô hình và bộ mã hóa nhãn...")
        model.save_model(MODEL_PATH)
        classes_list = le.classes_.tolist()
        with open(ENCODER_PATH, 'w') as f:
            json.dump(classes_list, f)
        print(f"Đã lưu mô hình vào file: '{MODEL_PATH}'")
        print(f"Đã lưu bộ mã hóa vào file: '{ENCODER_PATH}'")
    else:
        print("Không có dữ liệu để huấn luyện.")

if __name__ == '__main__':
    main()
