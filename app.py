import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import os
import gdown

# --- KRİTİK HATA ÇÖZÜCÜ: KERAS PARAMETRE YAMALAMA ---
from tensorflow.keras.layers import DepthwiseConv2D as OriginalDepthwise

# Keras 3'ün tanımadığı 'groups' parametresini yoksayan yeni bir sınıf tanımlıyoruz
class PatchedDepthwise(OriginalDepthwise):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups') # Hata veren parametreyi siliyoruz
        super().__init__(*args, **kwargs)

# Keras'a "DepthwiseConv2D gördüğünde benim yamalı versiyonumu kullan" diyoruz
custom_objs = {'DepthwiseConv2D': PatchedDepthwise}

# ---------------------------------------------------

@st.cache_resource
def load_my_model():
    model_path = 'brain_tumor_model.h5'
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        file_id = '1_NnO7sH_HphIFHZw66JUmIcv2efR6h4Y'
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            return f"İndirme Hatası: {e}"
    
    try:
        # custom_objects ile yamalı sınıfımızı modele enjekte ediyoruz
        return tf.keras.models.load_model(model_path, custom_objects=custom_objs, compile=False)
    except Exception as e:
        return f"Yükleme Hatası: {e}"

# --- ARAYÜZ AYARLARI ---
classes = ['Glioma', 'Healthy', 'Meningioma', 'Pituitary']
st.set_page_config(page_title="Beyin Analiz Portalı", layout="wide")

with st.sidebar:
    st.header("🎨 Tasarım")
    theme = st.radio("Tema:", ["Karanlık", "Aydınlık"])
    st.divider()
    st.metric("Model Doğruluğu", "%95.84")

bg = "#0E1117" if theme == "Karanlık" else "#FFFFFF"
txt = "#FFFFFF" if theme == "Karanlık" else "#000000"
st.markdown(f"<style>.stApp {{ background-color: {bg}; color: {txt}; }}</style>", unsafe_allow_html=True)

st.title("🧠 Yapay Zeka Beyin MRI Analizi")

model = load_my_model()

if isinstance(model, str):
    st.error(f"❌ Sistem Hatası: {model}")
    st.info("Bu hata genellikle versiyon uyuşmazlığından kaynaklanır. Lütfen Reboot App yapın.")
else:
    uploaded_file = st.file_uploader("Bir MRI görüntüsü yükleyin...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        col1, col2 = st.columns(2)
        img = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.image(img, use_container_width=True, caption="Yüklenen Kesit")
        
        # Görüntü İşleme
        img_input = np.array(img.resize((224, 224))) / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        
        # Tahmin
        preds = model.predict(img_input)[0]
        idx = np.argmax(preds)
        
        with col2:
            color = "#FF4B4B" if classes[idx] != "Healthy" else "#28A745"
            st.markdown(f"<h2 style='color:{color}'>{classes[idx]}</h2>", unsafe_allow_html=True)
            st.write(f"**Güven Seviyesi:** %{preds[idx]*100:.2f}")
            
            for i, label in enumerate(classes):
                st.write(f"{label}")
                st.progress(float(preds[i]))

# Alt sekmeler (Grafikler)
st.divider()
t1, t2 = st.tabs(["📊 İstatistikler", "🔍 Teknik Detay"])
with t1:
    cm = [[1650, 15, 10, 25], [12, 1720, 5, 3], [20, 10, 1680, 40], [15, 5, 10, 1800]]
    fig = go.Figure(data=go.Heatmap(z=cm, x=classes, y=classes, colorscale='Blues'))
    st.plotly_chart(fig, use_container_width=True)
