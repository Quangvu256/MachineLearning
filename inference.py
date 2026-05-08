"""Inference pipeline — predict single image end-to-end."""
import numpy as np
import xgboost as xgb
import json
import joblib
from PIL import Image
from features import get_combined_features
from config import MODEL_PATH, ENCODER_PATH, SCALER_PATH

class BirdClassifier:
    def __init__(self):
        self.model = xgb.XGBClassifier()
        self.model.load_model(MODEL_PATH)

        with open(ENCODER_PATH, 'r') as f:
            self.classes = json.load(f)

        self.scaler = joblib.load(SCALER_PATH)

    def predict(self, pil_image: Image.Image) -> tuple[str, dict]:
        """
        Returns: (predicted_class_name, {class_name: probability})
        """
        features = get_combined_features(pil_image)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict probabilities
        proba = self.model.predict_proba(features_scaled)[0]
        predicted_idx = np.argmax(proba)
        predicted_class = self.classes[predicted_idx]

        # Map probabilities to class names
        prob_dict = {self.classes[i]: float(proba[i]) for i in range(len(self.classes))}

        return predicted_class, prob_dict
