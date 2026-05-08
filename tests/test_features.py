"""Unit tests cho feature extraction pipeline."""
import numpy as np
from PIL import Image
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features import get_combined_features, extract_hog_features, extract_lbp_features, extract_color_histogram

def test_feature_dimension_consistency():
    """Verify feature vector length nhất quán qua nhiều image sizes."""
    sizes = [(64, 64), (128, 128), (256, 256), (100, 200)]
    dims = []
    for size in sizes:
        img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
        features = get_combined_features(img)
        dims.append(len(features))
    assert len(set(dims)) == 1, f"Feature dimensions inconsistent: {dims}"

def test_grayscale_input():
    """Verify pipeline handles grayscale images."""
    gray = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode='L')
    features = get_combined_features(gray.convert('RGB'))
    assert features.shape[0] > 0

def test_rgba_input():
    """Verify pipeline handles RGBA images."""
    rgba = Image.fromarray(np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8), mode='RGBA')
    features = get_combined_features(rgba.convert('RGB'))
    assert features.shape[0] > 0
