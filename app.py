import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import cv2

# Must be the first Streamlit command
st.set_page_config(page_title="GTSRB Traffic Sign Classifier", layout="centered")

# Load model (with Streamlit caching)
@st.cache_resource
def load_cnn_model():
    return load_model("best_model.h5")

model = load_cnn_model()

# Load label binarizer (for class names)
@st.cache_resource
def load_label_binarizer():
    with open("label_binarizer.pkl", "rb") as f:
        return pickle.load(f)

label_binarizer = load_label_binarizer()
class_names = label_binarizer.classes_

def preprocess_image(image_data):
    """
    Preprocess uploaded image:
    - Converts to RGB
    - Resizes to (64, 64)
    - Normalizes pixel values to [0, 1]
    - Adds batch dimension
    """
    image = Image.open(image_data).convert("RGB")
    image_np = np.array(image)

    # Resize using OpenCV to (64, 64) for consistency with training
    image_resized = cv2.resize(image_np, (64, 64))

    # Normalize and expand dims
    image_normalized = image_resized.astype("float32") / 255.0
    return np.expand_dims(image_normalized, axis=0)

# Streamlit UI
st.title("🚦 German Traffic Sign Classifier")
st.write("Upload an image of a German traffic sign to classify it:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)
    
    predicted_class_idx = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx] * 100

    # Debug: Show prediction confidence for all classes (optional)
    with st.expander("🔬 Prediction Confidence for All Classes"):
        for i, prob in enumerate(prediction[0]):
            st.write(f"{class_names[i]}: {prob * 100:.2f}%")

    # Output prediction
    st.markdown(f"### 🧠 Predicted Class: **{predicted_class_name}**")
    st.markdown(f"#### 🔍 Confidence: **{confidence:.2f}%**")
