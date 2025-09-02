import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

st.title("🌾 Rice Classifier (CNN Model)")

# ✅ تحقق إن الموديل موجود
if os.path.exists("rice_cnn_model.h5"):
    model = tf.keras.models.load_model("rice_cnn_model.h5")
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Model file 'rice_cnn_model.h5' not found. Please put it in the same folder as app.py")
    st.stop()

# الأصناف (غيريها حسب تدريبك)
class_names = ["Basmati", "Jasmine", "Arborio", "Ipsala", "Karacadag"]

# رفع صورة
uploaded_file = st.file_uploader("Upload an image of rice grain", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ جهزي الصورة للحجم المناسب (غيري 224 لو دربتي على حجم مختلف)
    img_size = (64, 64)
    img_array = image.resize(img_size)
    img_array = np.array(img_array) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ prediction
  # ✅ prediction
try:
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # 👌 عرض النتيجة النصية
    st.success(f"✅ Prediction: {class_names[predicted_class]}")

    # 🎨 عرض الاحتمالات كـ Bar Chart مع تمييز الـ predicted class
    colors = ["skyblue"] * len(class_names)
    colors[predicted_class] = "limegreen"  # ✅ اللون الأخضر للـ predicted class

    fig, ax = plt.subplots()
    ax.bar(class_names, prediction[0], color=colors)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    plt.xticks(rotation=45)
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Error during prediction: {e}")
