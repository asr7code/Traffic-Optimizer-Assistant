import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import cv2

# Set Streamlit config FIRST
st.set_page_config(page_title="Traffic Sign Voice Alert", layout="centered")

# Load model and label binarizer with caching
@st.cache_resource
def load_cnn_model():
    return load_model("best_model.h5")

@st.cache_resource
def load_label_binarizer():
    with open("label_binarizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_cnn_model()
label_binarizer = load_label_binarizer()
class_names = label_binarizer.classes_

def preprocess_image(image_data):
    image = Image.open(image_data).convert("RGB")
    image_np = np.array(image)
    image_bgr = image_np[..., ::-1]
    image_bgr = cv2.resize(image_bgr, (64, 64))
    image_bgr = image_bgr.astype("float32") / 255.0
    return np.expand_dims(image_bgr, axis=0)

def auto_speak_js(label):
    safe_label = str(label).replace('"', '\\"')
    st.components.v1.html(f"""
    <script>
        const msg = new SpeechSynthesisUtterance("Caution! {safe_label}");
        window.speechSynthesis.cancel();  // Cancel any existing speech
        window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

# Streamlit UI
st.title("🚦 Traffic Sign Classifier with Voice Alert")

uploaded_file = st.file_uploader("Upload a traffic sign image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Predict
    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    # Display prediction
    st.markdown(f"### 🧠 Predicted Sign: **{predicted_label}**")
    st.markdown(f"#### 🔍 Confidence: **{confidence:.2f}%**")

    # Trigger speech alert automatically
    auto_speak_js(predicted_label)
