import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import os
import gdown

# 1. HATA ÇÖZÜCÜ: Keras 3 ve Versiyon Uyumsuzlukları İçin
# Hem 'groups' hatasını hem de 'standardize_shape' sorunlarını minimize eder
custom_objs = {
    'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D,
    'Functional': tf.keras.models.Model,
    'Sequential': tf.keras.models.Sequential
}

# 2. MODEL İNDİRME VE YÜKLEME
@st.cache_resource
def load_my_model():
    model_path = 'brain_tumor_model.h5'
    # Dosya yoksa veya bozuksa Drive'dan indir
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        file_id = '1_NnO7sH_HphIFHZw66JUmIcv2efR6h4Y'
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            return f"İndirme Hatası: {e}"
    
    try:
        # compile=False diyerek modelin eğitim ayarlarını (optimizer vb.) yüklemiyoruz, 
        # bu sayede versiyon farklarından kaynaklanan hataları bypass ediyoruz.
        return tf.keras.models.load_model(model_path, custom_objects=custom_objs, compile=False)
    except Exception as e:
        return f"Yükleme Hatası: {e}"

# 3. TEMEL TANIMLAMALAR
classes = ['Glioma', 'Healthy', 'Meningioma', 'Pituitary']

# 4. SAYFA AYARLARI
st.set_page_config(page_title="Beyin Tümörü Analizi", layout="wide")

# Kenar Çubuğu (Sidebar)
with st.sidebar:
    st.header("🎨 Görünüm")
    theme_choice = st.radio("Tema Seçiniz:", ["Karanlık (Dark)", "Aydınlık (Light)"])
    st.divider()
    st.subheader("📊 Performans")
    st.metric(label="Accuracy", value="%95.84")

# Dinamik CSS
if theme_choice == "Karanlık (Dark)":
    bg, txt, card = "#0E1117", "#FFFFFF", "#161B22"
else:
    bg, txt, card = "#FFFFFF", "#000000", "#F0F2F6"

st.markdown(f"<style>.stApp {{ background-color: {bg}; color: {txt}; }}</style>", unsafe_allow_html=True)
st.title("🧠 Beyin Tümörü Analiz Sistemi")

# Model Yükleme
model = load_my_model()

if isinstance(model, str):
    st.error(f"⚠️ Kritik Hata: {model}")
    st.info("Lütfen 'Manage App' kısmından Reboot ederek tekrar deneyin.")
else:
    # 5. ANA ANALİZ EKRANI
    uploaded_file = st.file_uploader("MRI Görüntüsü Seçiniz...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        img = Image.open(uploaded_file).convert("RGB")
        
        with col1:
            st.image(img, caption="Yüklenen Görüntü", use_container_width=True)
        
        # Ön İşleme (Input Shape Kontrolü)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Tahmin
        predictions = model.predict(img_array)[0]
        result_idx = np.argmax(predictions)
        
        with col2:
            st.success(f"Teşhis: **{classes[result_idx]}**")
            st.write(f"Güven Oranı: %{predictions[result_idx]*100:.2f}")
            
            # Olasılık Çubukları
            for i in range(len(classes)):
                st.write(f"{classes[i]}")
                st.progress(float(predictions[i]))

# 6. ALT TABLAR
t1, t2 = st.tabs(["📊 Analiz", "💻 Sistem"])
with t1:
    st.write("Model performansı ve Confusion Matrix verileri burada yer almaktadır.")
with t2:
    st.code("Model: MobileNetV2 tabanlı Transfer Learning\nInput: 224x224x3")
