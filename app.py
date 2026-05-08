import streamlit as st
import xgboost as xgb
import json
import numpy as np
from PIL import Image
import cv2
import os

# 1. Cấu hình trang để responsive (fix với màn hình hiển thị)
st.set_page_config(page_title="Demo Nhận Diện Chim", layout="wide")

# 2. Định nghĩa 5 lớp chim (dựa trên cấu trúc thư mục của bạn)
BIRD_CLASSES = [
    "016.Painted_Bunting",
    "073.Blue_Jay",
    "089.Hooded_Merganser",
    "096.Hooded_Oriole",
    "147.Least_Tern"
]

from inference import BirdClassifier

@st.cache_resource
def load_classifier():
    return BirdClassifier()

def main():
    st.title("🦅 Ứng dụng AI Nhận diện 5 Lớp Chim")
    
    st.markdown("### Các phân lớp hỗ trợ nhận diện:")
    cols = st.columns(5)
    for i, bird in enumerate(BIRD_CLASSES):
        cols[i].info(bird)

    # Load model
    try:
        classifier = load_classifier()
        st.success("Tải mô hình (Model) mặc định thành công!")
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}\n(Vui lòng chạy `python train.py` trước nếu chưa có model)")
        return

    st.markdown("---")
    
    # Upload hình ảnh
    uploaded_file = st.file_uploader("📥 Chọn ảnh chim để dự đoán...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Layout hiển thị ảnh và kết quả
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Hiện ảnh, use_container_width=True giúp ảnh tự động scale theo màn hình (responsive)
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
            
        with col2:
            st.subheader("🎯 Cập nhật Kết quả Dự đoán")
            with st.spinner("Đang xử lý và phân tích ảnh..."):
                try:
                    predicted_class, probabilities = classifier.predict(image)
                    st.success(f"🎯 Dự đoán: **{predicted_class}**")
                    st.bar_chart(probabilities)
                except Exception as e:
                    st.error(f"Lỗi phân tích: {e}")

if __name__ == "__main__":
    main()
