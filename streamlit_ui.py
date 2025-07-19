import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Constants
IMAGE_SIZE = (256, 256)  # ✅ Match your model's expected input shape
MODEL_PATH = "crop-disease_model.h5"
DATA_DIR = os.path.join(os.path.expanduser("~"), "Documents", "machine_learning", "crop_diseases_CNN", "train")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Automatically get class names from training directory
CLASS_NAMES = sorted(os.listdir(DATA_DIR))  # Ensure alphabetical order
st.sidebar.success(f"✅ Model loaded with {len(CLASS_NAMES)} classes")

# Streamlit UI
st.title("🌿 Crop Disease Detector")
st.write("Upload a crop image to detect the disease category")

uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize(IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)[0]
    pred_index = np.argmax(prediction)
    pred_class = CLASS_NAMES[pred_index]
    confidence = prediction[pred_index]

    # Display result
    st.subheader("🔍 Prediction")
    st.success(f"Predicted: **{pred_class}**")
    st.write(f"Confidence: `{confidence:.2%}`")

    # Optional: Show probabilities for all classes
    with st.expander("See all class probabilities"):
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_name}: {prediction[i]:.2%}")
