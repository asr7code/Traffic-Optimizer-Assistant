import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="Traffic Sign Classifier", layout="centered")

# Function to load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5")
    return model

# Function to load class names from CSV
@st.cache_data
def load_class_names():
    df = pd.read_csv("signnames.csv")
    return dict(zip(df['ClassId'], df['SignName']))

# Preprocess image for prediction
def preprocess_image(image):
    image = image.resize((30, 30))  # Resize to 30x30
    image = np.array(image)
    if image.shape[-1] == 4:  # Remove alpha channel if present
        image = image[:, :, :3]
    image = image / 255.0  # Normalize
    image = image[np.newaxis, ...]  # Add batch dimension
    return image

# Load model and class names
model = load_model()
class_names = load_class_names()

# Streamlit UI
st.title("🚦 Traffic Sign Classifier")
st.markdown("Upload a traffic sign image and get its prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("⏳ Predicting...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    class_name = class_names.get(class_index, "Unknown")

    st.success(f"### 🧠 Prediction: {class_name}")
    st.info(f"🔢 Class ID: {class_index} | 🔍 Confidence: {confidence:.2f}")
