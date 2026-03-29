import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import gdown
from streamlit_option_menu import option_menu

# --- 1. MODEL YÜKLEME (SENİN DRIVE LİNKİNLE GÜNCELLENDİ) ---
@st.cache_resource
def load_tumor_model():
    # Senin paylaştığın linkten alınan ID
    file_id = '1_NnO7sH_HphIFHZw66JUmIcv2efR6h4Y' 
    model_path = 'brain_tumor_model.h5'
    
    if not os.path.exists(model_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Model indirilirken hata oluştu: {e}")
    
    return tf.keras.models.load_model(model_path)

model = load_tumor_model()
class_names = ['Glioma', 'Healthy (Sağlıklı)', 'Meningioma', 'Pituitary (Hipofiz)']

# --- 2. GÖRSEL DOĞRULAMA (MRI KONTROLÜ) ---
def is_mri(image):
    img_array = np.array(image.convert('L'))
    edges = cv2.Canny(img_array, 100, 200)
    return np.mean(edges) < 55 

# --- 3. ARAYÜZ TASARIMI ---
st.set_page_config(page_title="NeuroScan AI", layout="wide")

with st.sidebar:
    selected = option_menu(
        "Navigasyon", ["Analiz Paneli", "Proje Detayları"],
        icons=['cpu', 'info-circle'], menu_icon="cast", default_index=0,
    )

if selected == "Analiz Paneli":
    st.title("🧠 NeuroScan: Beyin Tümörü Karar Destek Sistemi")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Bir MRI görüntüsü seçin...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Analiz Edilen Görüntü", use_container_width=True)
        
        with col2:
            if not is_mri(image):
                st.error("⚠️ Bu bir beyin MRI görüntüsü gibi görünmüyor. Lütfen geçerli bir kesit yükleyin.")
            else:
                with st.spinner('Yapay Zeka Modeli Analiz Ediyor...'):
                    # Ön İşleme
                    size = (224, 224)
                    processed_image = ImageOps.fit(image, size, Image.LANCZOS)
                    img_array = np.asarray(processed_image) / 255.0
                    img_reshape = img_array[np.newaxis, ...]
                    
                    # Tahmin
                    prediction = model.predict(img_reshape)
                    result_idx = np.argmax(prediction)
                    confidence = prediction[0][result_idx]
                    
                    # --- GÜVEN EŞİĞİ (EZBER KONTROLÜ) ---
                    if confidence < 0.85:
                        st.warning(f"🔍 **Düşük Güven: %{confidence*100:.2f}**")
                        st.info("Model bu görüntüden tam emin olamadı. Bu durum, görüntünün net olmamasından veya nadir bir vakadan kaynaklanabilir.")
                    else:
                        st.success(f"🎯 **Teşhis: {class_names[result_idx]}**")
                        st.metric("Güven Seviyesi", f"%{confidence*100:.2f}")
                        
                        # Olasılık Dağılımı
                        for i in range(4):
                            st.write(f"{class_names[i]}:")
                            st.progress(float(prediction[0][i]))

else:
    st.title("ℹ️ Proje Hakkında")
    st.write("Bu çalışma, MobileNetV2 mimarisi üzerine inşa edilmiştir.")
    st.success("Doğrulama Başarısı: %91.31")
    st.info("Data Augmentation kullanılarak overfitting (ezberleme) engellenmiştir.")
