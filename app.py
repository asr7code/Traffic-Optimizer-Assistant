import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import cv2

# Set Streamlit page configuration FIRST
st.set_page_config(page_title="GTSRB Traffic Sign Classifier", layout="centered")

# Load the trained CNN model
@st.cache_resource
def load_cnn_model():
    return load_model("best_model.h5")

model = load_cnn_model()

# Load the label binarizer for class names
@st.cache_resource
def load_label_binarizer():
    with open("label_binarizer.pkl", "rb") as f:
        return pickle.load(f)

label_binarizer = load_label_binarizer()
class_names = label_binarizer.classes_

# Function to preprocess uploaded image
def preprocess_image(image_data):
    image = Image.open(image_data).convert("RGB")
    image_np = np.array(image)
    image_bgr = image_np[..., ::-1]  # Convert RGB to BGR
    image_bgr = cv2.resize(image_bgr, (64, 64))
    image_bgr = image_bgr.astype("float32") / 255.0
    return np.expand_dims(image_bgr, axis=0)

# Function to trigger browser speech
def speak_label(label):
    safe_label = label.replace('"', '\\"')  # Escape quotes for JavaScript
    st.markdown(
        f"""
        <script>
            var msg = new SpeechSynthesisUtterance("Caution. {safe_label}");
            window.speechSynthesis.speak(msg);
        </script>
        """,
        unsafe_allow_html=True
    )

# Streamlit UI
st.title("🚦 German Traffic Sign Classifier with Voice Alert")
st.write("Upload an image of a German traffic sign to classify it and hear its name!")

uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    predicted_label = class_names[predicted_class]
    confidence = prediction[0][predicted_class] * 100

    st.markdown(f"### 🧠 Predicted Class: **{predicted_label}**")
    st.markdown(f"#### 🔍 Confidence: **{confidence:.2f}%**")

    # Voice alert
    speak_label(predicted_label)
