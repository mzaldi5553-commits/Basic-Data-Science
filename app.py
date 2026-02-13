import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Gaji Lulusan Vokasi", layout="centered")

# --- LOAD MODEL DAN SCALER ---
# Pastikan file .pkl berada di folder yang sama dengan script ini
@st.cache_resource
def load_assets():
    model = joblib.load('model_gaji.pkl')
    scaler = joblib.load('scaler.pkl')
    # Muat list kolom fitur saat training agar urutannya konsisten
    # Jika tidak disimpan, Anda harus mendefinisikannya secara manual
    features_columns = joblib.load('features_columns.pkl') 
    return model, scaler, features_columns

try:
    model, scaler, model_features = load_assets()
except Exception as e:
    st.error(f"Gagal memuat model/scaler: {e}")
    st.stop()

# --- UI APLIKASI ---
st.title("💰 Prediksi Gaji Awal Lulusan Vokasi")
st.markdown("Masukkan data alumni di bawah ini untuk mengestimasi gaji awal.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        usia = st.number_input("Usia (Tahun)", min_value=17, max_value=60, value=20)
        durasi_pelatihan = st.number_input("Durasi Pelatihan (Bulan)", min_value=1, max_value=36, value=6)
        nilai_ujian = st.slider("Nilai Ujian Akhir", 0, 100, 85)
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

    with col2:
        pendidikan = st.selectbox("Pendidikan Terakhir", ["SMA/SMK", "Diploma", "Sarjana"])
        jurusan = st.selectbox("Jurusan Pelatihan", ["Teknik", "IT", "Bisnis", "Desain", "Kesehatan"])
        status_bekerja = st.radio("Memiliki Pengalaman Kerja?", ["Ya", "Tidak"])

    submit = st.form_submit_button("Prediksi Gaji")

# --- PREPROCESSING & PREDIKSI ---
if submit:
    # 1. Buat DataFrame dari input
    input_data = pd.DataFrame({
        'usia': [usia],
        'durasi_pelatihan': [durasi_pelatihan],
        'nilai_ujian': [nilai_ujian],
        'pendidikan': [pendidikan],
        'jurusan': [jurusan],
        'jenis_kelamin': [1 if jenis_kelamin == "Laki-laki" else 0],
        'status_bekerja': [1 if status_bekerja == "Ya" else 0]
    })

    # 2. One-Hot Encoding (Sesuaikan dengan metode saat training)
    # Ini harus menghasilkan kolom yang sama persis dengan model_features
    input_encoded = pd.get_dummies(input_data)
    
    # Menyelaraskan kolom input dengan kolom saat training (menambah kolom 0 jika hilang)
    for col in model_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    # Memastikan urutan kolom tepat sama
    input_final = input_encoded[model_features]

    # 3. Scaling
    input_scaled = scaler.transform(input_final)

    # 4. Prediksi
    prediction = model.predict(input_scaled)
    gaji_format = f"Rp {prediction[0]:,.0f}".replace(",", ".")

    # --- TAMPILKAN HASIL ---
    st.success(f"### Estimasi Gaji Awal: {gaji_format}")
    
    with st.expander("Lihat Detail Input"):
        st.write("Data yang dikirim ke model (setelah preprocessing):")
        st.dataframe(input_final)

# --- FOOTER ---
st.divider()
st.caption("Catatan: Hasil ini adalah estimasi berdasarkan data historis pelatihan vokasi.")
