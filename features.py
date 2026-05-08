"""Feature extraction module — single source of truth cho train & inference."""
import numpy as np
from PIL import Image
import cv2
from skimage.feature import hog, local_binary_pattern
from config import IMAGE_SIZE, HOG_PARAMS, LBP_PARAMS, HSV_BINS

def extract_hog_features(pil_image):
    gray = np.array(pil_image.convert('L').resize(IMAGE_SIZE))
    # Dùng PIL resize (uint8) thay vì skimage resize (float) → nhất quán
    features = hog(gray, **HOG_PARAMS, visualize=False)
    return features

def extract_color_histogram(pil_image):
    img_resized = np.array(pil_image.resize(IMAGE_SIZE))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HSV_BINS, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp_features(pil_image):
    gray = np.array(pil_image.convert('L').resize(IMAGE_SIZE))
    n_points = LBP_PARAMS['n_points']
    lbp = local_binary_pattern(gray, n_points, LBP_PARAMS['radius'], LBP_PARAMS['method'])
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist

def get_combined_features(pil_image):
    """Trả về feature vector hợp nhất. Dùng chung cho train & inference."""
    return np.concatenate([
        extract_hog_features(pil_image),
        extract_color_histogram(pil_image),
        extract_lbp_features(pil_image),
    ])
