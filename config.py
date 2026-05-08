import os

IMAGE_SIZE = (128, 128)

# Auto-detect device
def get_xgb_device():
    """Trả về 'cuda' nếu có GPU, ngược lại 'cpu'"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        if result.returncode == 0:
            return 'cuda'
    except Exception:
        pass
    return 'cpu'

DEVICE = get_xgb_device()

# Feature params (single source of truth)
HOG_PARAMS = {'pixels_per_cell': (16, 16), 'cells_per_block': (2, 2)}
LBP_PARAMS = {'radius': 3, 'n_points': 24, 'method': 'uniform'}
HSV_BINS = [8, 8, 8]

# Paths
MODEL_DIR = 'models'
MODEL_PATH = f'{MODEL_DIR}/xgboost_model.json'
ENCODER_PATH = f'{MODEL_DIR}/label_encoder.json'
SCALER_PATH = f'{MODEL_DIR}/feature_scaler.pkl'
PROCESSED_DATA_DIR = 'ML'

# Training
SEED = 42
TEST_SIZE = 0.25
N_AUGMENTATIONS = 3
CV_FOLDS = 3
N_SEARCH_ITER = 30
EARLY_STOPPING_ROUNDS = 30
