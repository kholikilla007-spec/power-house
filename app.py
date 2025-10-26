
import streamlit as st  
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- Page setup ---
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image to predict the tumor type")

# --- Load model ---
MODEL_PATH = r"keras_model.h5"  # update this path to where your model is stored
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# --- Define class names ---
class_names = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}

# --- File uploader ---
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # --- Preprocess ---
    img = img.resize((224, 224))  # Teachable Machine usually uses 224x224
    img_array = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # --- Predict ---
    predictions = model.predict(img_batch)
    pred_index = np.argmax(predictions[0])
    pred_class = class_names[pred_index]
    confidence = float(predictions[0][pred_index]) * 100

    # --- Display ---
    st.markdown(f"### 🩺 Prediction: **{pred_class}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
