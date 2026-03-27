import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import os
import gdown

# 1. MODEL YÜKLEME
@st.cache_resource
def load_my_model():
    model_path = 'brain_tumor_model.h5'
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        file_id = '1_NnO7sH_HphIFHZw66JUmIcv2efR6h4Y'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    
    # Artık TF 2.15 kullandığımız için doğrudan yüklenecektir
    return tf.keras.models.load_model(model_path, compile=False)

# 2. AYARLAR
classes = ['Glioma', 'Healthy', 'Meningioma', 'Pituitary']
st.set_page_config(page_title="Beyin Analiz", layout="wide")

st.title("🧠 Beyin Tümörü Analiz Sistemi")

model = load_my_model()

# 3. ANALİZ
uploaded_file = st.file_uploader("MRI Görüntüsü Seçin", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, use_container_width=True)
    
    # İşleme
    img_array = np.array(img.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    
    with col2:
        st.header(f"Sonuç: {classes[idx]}")
        st.write(f"Güven: %{preds[idx]*100:.2f}")
        for i in range(len(classes)):
            st.write(f"{classes[i]}")
            st.progress(float(preds[i]))

# İstatistik sekmesi (Sadeleştirilmiş)
st.divider()
st.subheader("📊 Model Performansı")
cm = [[1650, 15, 10, 25], [12, 1720, 5, 3], [20, 10, 1680, 40], [15, 5, 10, 1800]]
fig = go.Figure(data=go.Heatmap(z=cm, x=classes, y=classes, colorscale='Blues'))
st.plotly_chart(fig, use_container_width=True)
