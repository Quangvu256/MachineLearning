import os
import tarfile
import urllib.request
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm

# --- CẤU HÌNH ---
# Nếu thư mục này không tồn tại hoặc thiếu metadata, script sẽ tự tải dataset về DATA_BASE_DIR
ROOT_DIR = os.environ.get('CUB_DATA_DIR', 'data/CUB_200_2011/CUB_200_2011')
OUTPUT_DIR = 'CUB_200_processed'

# Link và nơi lưu khi auto-download
DATA_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
DATA_BASE_DIR = "data"  # file .tgz và thư mục giải nén sẽ nằm trong đây

# --- HÀM TẢI & GIẢI NÉN ĐƠN GIẢN ---
def download_and_extract_cub(base_dir=DATA_BASE_DIR, url=DATA_URL):
    """
    Tải file .tgz của CUB_200_2011 và giải nén vào base_dir.
    Trả về đường dẫn ROOT_DIR: <base_dir>/CUB_200_2011
    """
    os.makedirs(base_dir, exist_ok=True)
    tgz_path = os.path.join(base_dir, "CUB_200_2011.tgz")

    if not os.path.exists(tgz_path):
        print("Đang tải CUB_200_2011 (~1.1GB), vui lòng chờ...")
        urllib.request.urlretrieve(url, tgz_path)
        print(f"Đã tải: {tgz_path}")
    else:
        print("Đã có file nén, bỏ qua bước tải.")

    print("Đang giải nén...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, path=base_dir)

    extracted_root = os.path.join(base_dir, "CUB_200_2011")
    if not os.path.isdir(extracted_root):
        raise RuntimeError("Giải nén không thành công: không thấy thư mục CUB_200_2011.")
    print(f"Hoàn tất! Dataset ở: {extracted_root}")
    return extracted_root

# --- CÁC HÀM HỖ TRỢ ---
def get_data_metadata(root_dir):
    """
    Đọc các file metadata của bộ CUB-200-2011 và gộp chúng lại.
    - images.txt: ID và đường dẫn tương đối của ảnh.
    - image_class_labels.txt: ID ảnh và nhãn lớp (loài chim).
    - classes.txt: ID lớp và tên lớp (tên loài chim).
    - bounding_boxes.txt: ID ảnh và tọa độ hộp giới hạn.

    Returns:
        pandas.DataFrame: DataFrame tổng hợp cho mỗi ảnh.
    """
    print("Đang đọc các file metadata...")
    # sep=r"\s+" để an toàn với mọi số lượng khoảng trắng
    images_df = pd.read_csv(os.path.join(root_dir, 'images.txt'),
                            sep=r'\s+', names=['img_id', 'filepath'], engine='python')
    labels_df = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'),
                            sep=r'\s+', names=['img_id', 'label_id'], engine='python')
    classes_df = pd.read_csv(os.path.join(root_dir, 'classes.txt'),
                            sep=r'\s+', names=['label_id', 'class_name'], engine='python')
    bounding_boxes_df = pd.read_csv(os.path.join(root_dir, 'bounding_boxes.txt'),
                                    sep=r'\s+', names=['img_id', 'x', 'y', 'width', 'height'], engine='python')

    data = images_df.merge(labels_df, on='img_id')
    data = data.merge(classes_df, on='label_id')
    data = data.merge(bounding_boxes_df, on='img_id')

    print(f"Đã đọc xong thông tin cho {len(data)} ảnh.")
    return data

def process_and_save_single_image(image_info_row, root_dir, output_dir):
    """
    Xử lý MỘT ảnh: đọc, cắt theo bbox và lưu.
    """
    try:
        source_path = os.path.join(root_dir, 'images', image_info_row['filepath'])
        x, y, w, h = image_info_row['x'], image_info_row['y'], image_info_row['width'], image_info_row['height']
        class_name = image_info_row['class_name']

        # Thư mục đích dạng: 'CUB_200_processed/001.Black_footed_Albatross/'
        target_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        image_filename = os.path.basename(image_info_row['filepath'])
        target_path = os.path.join(target_class_dir, image_filename)

        # Mở ảnh, clamp bbox về biên ảnh đơn giản
        with Image.open(source_path).convert('RGB') as img:
            img_w, img_h = img.size
            # Chuyển bbox sang int và chặn ra ngoài biên
            left = max(0, int(round(x)))
            top = max(0, int(round(y)))
            right = min(img_w, int(round(x + w)))
            bottom = min(img_h, int(round(y + h)))
            if right <= left or bottom <= top:
                # bbox lỗi -> fallback lưu ảnh gốc
                cropped_img = img
            else:
                cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(target_path)

    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {source_path}: {e}")

# --- HÀM CHÍNH ---
def main():
    """
    Điều phối toàn bộ quá trình:
    - Kiểm tra dataset, nếu thiếu -> tải & giải nén
    - Đọc metadata
    - Cắt ảnh theo bbox, lưu vào OUTPUT_DIR
    """
    # Tự động tải nếu thiếu
    global ROOT_DIR
    need_download = (not os.path.isdir(ROOT_DIR) or
                     not os.path.exists(os.path.join(ROOT_DIR, "images.txt")))
    if need_download:
        print(f"Không tìm thấy dataset tại '{ROOT_DIR}'. Tự động tải về...")
        ROOT_DIR = download_and_extract_cub()  # -> data/CUB_200_2011

    # Bước 1: Đọc metadata
    all_metadata = get_data_metadata(ROOT_DIR)

    # Bước 2: Tạo thư mục đầu ra
    print(f"Tạo thư mục đầu ra tại: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Bước 3: Xử lý song song
    print(f"\nBắt đầu tiền xử lý và cắt {len(all_metadata)} ảnh bằng tất cả các nhân CPU...")
    Parallel(n_jobs=-1)(
        delayed(process_and_save_single_image)(row, ROOT_DIR, OUTPUT_DIR)
        for _, row in tqdm(all_metadata.iterrows(), total=len(all_metadata))
    )

    print("\n----------------------------------------------------")
    print("HOÀN TẤT TIỀN XỬ LÝ DỮ LIỆU!")
    print(f"Tất cả các ảnh đã được cắt và lưu vào thư mục: '{OUTPUT_DIR}'")
    print("----------------------------------------------------")

if __name__ == '__main__':
    main()
