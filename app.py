import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import cv2

# Must be the first Streamlit command
st.set_page_config(page_title="GTSRB Traffic Sign Classifier", layout="centered")

# Load the trained model
@st.cache_resource
def load_cnn_model():
    return load_model("best_model.h5")

model = load_cnn_model()

# Load the label binarizer
@st.cache_resource
def load_label_binarizer():
    with open("label_binarizer.pkl", "rb") as f:
        return pickle.load(f)

label_binarizer = load_label_binarizer()
class_names = label_binarizer.classes_

# Function to preprocess image
def preprocess_image(image_data):
    image = Image.open(image_data).convert("RGB")
    image_np = np.array(image)
    image_bgr = image_np[..., ::-1]  # RGB to BGR
    image_bgr = cv2.resize(image_bgr, (64, 64))
    image_bgr = image_bgr.astype("float32") / 255.0
    return np.expand_dims(image_bgr, axis=0)

# Function to speak the predicted label using JS
def speak_label(label):
    safe_label = str(label).replace('"', '\\"')
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
st.write("Upload an image of a German traffic sign to classify and hear the prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_label = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    # Display prediction and confidence
    st.markdown(f"### 🧠 Predicted Sign: **{predicted_label}**")
    st.markdown(f"#### 🔍 Confidence: **{confidence:.2f}%**")

    # Speak the prediction
    speak_label(predicted_label)
