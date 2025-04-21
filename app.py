import streamlit as st # type: ignore
import os
from PIL import Image

# Inisialisasi folder
DATA_DIR = "dataset/raw"
os.makedirs(DATA_DIR, exist_ok=True)

st.title("EthniScan Dataset Manager")

uploaded_files = st.file_uploader("Upload gambar wajah", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file)
        save_path = os.path.join(DATA_DIR, file.name)
        img.save(save_path)
    st.success(f"{len(uploaded_files)} gambar berhasil di-upload ke `{DATA_DIR}`.")