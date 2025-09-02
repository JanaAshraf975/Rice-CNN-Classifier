import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

st.title("🌾 Rice Classifier (CNN Model)")

# ✅ تحقق إن الموديل موجود
model_path = "rice_cnn_model.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    st.success("✅ Model loaded successfully!")
else:
    st.error(f"❌ Model file '{model_path}' not found. Please put it in the same folder as app.py")
    st.stop()

# ✅ Automatically fetch class names from model output
num_classes = model.output_shape[-1]
# Edit this list if you know the correct class names
class_names = ['Karacadag', 'Basmati', 'Jasmine', 'Arborio', 'Ipsala']  

if len(class_names) != num_classes:
    st.error(f"❌ Number of class names ({len(class_names)}) does not match model output ({num_classes})")
    st.stop()

st.write(f"Loaded class names: {class_names}")

# Upload image
uploaded_file = st.file_uploader("Upload an image of rice grain", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize & normalize
    img_size = (64, 64)
    img_array = np.array(image.resize(img_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        st.success(f"✅ Prediction: {class_names[predicted_class]}")

        # Display probabilities
        colors = ["skyblue"] * len(class_names)
        colors[predicted_class] = "limegreen"

        fig, ax = plt.subplots()
        ax.bar(class_names, prediction[0], color=colors)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probabilities")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
