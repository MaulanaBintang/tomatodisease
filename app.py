import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# ==== Load model ====
MODEL_PATH = "model_daun_tomat_mobilenetv2.h5"
model = load_model(MODEL_PATH)

# ==== Parameter ====
IMG_SIZE = 224
class_names = ['Sehat', 'Penyakit1', 'Penyakit2']  # Ganti sesuai kelasmu

# ==== Fungsi Prediksi ====
def predict(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    confidence = np.max(preds)
    return class_names[pred_idx], confidence

# ==== UI Streamlit ====
st.set_page_config(page_title="Deteksi Daun Tomat", layout="centered")
st.title("Deteksi Penyakit Daun Tomat")
st.write("Upload gambar atau ambil dari kamera untuk mendeteksi penyakit pada daun tomat.")

# Pilihan input gambar
tab_upload, tab_camera = st.tabs(["Upload Gambar", "Gunakan Kamera"])

with tab_upload:
    uploaded_file = st.file_uploader("Pilih gambar daun tomat...", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image_data = Image.open(uploaded_file).convert("RGB")
        st.image(image_data, caption="Gambar yang Diunggah", use_column_width=True)
        if st.button("Prediksi Gambar"):
            label, conf = predict(image_data)
            st.success(f"Hasil Prediksi: **{label}** ({conf*100:.2f}%)")

with tab_camera:
    camera_image = st.camera_input("Ambil gambar daun tomat:")
    if camera_image is not None:
        image_data = Image.open(camera_image).convert("RGB")
        st.image(image_data, caption="Gambar dari Kamera", use_column_width=True)
        if st.button("Prediksi Kamera"):
            label, conf = predict(image_data)
            st.success(f"Hasil Prediksi: **{label}** ({conf*100:.2f}%)")