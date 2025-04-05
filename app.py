import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import cv2
import streamlit.components.v1 as components

# Must be first Streamlit command
st.set_page_config(page_title="GTSRB Traffic Sign Classifier", layout="centered")

# Load the model
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

# Voice alert using browser
def speak_js(message):
    js = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{message}");
        msg.lang = "en-US";
        msg.rate = 1;
        window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js)

def preprocess_image(image_data):
    image = Image.open(image_data).convert("RGB")
    image_np = np.array(image)
    image_bgr = image_np[..., ::-1]
    image_bgr = cv2.resize(image_bgr, (64, 64))
    image_bgr = image_bgr.astype("float32") / 255.0
    return np.expand_dims(image_bgr, axis=0)

# UI
st.title("🚦 German Traffic Sign Classifier")
st.write("Upload an image of a traffic sign. The app will classify it and speak the result.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)
    
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    predicted_label = class_names[predicted_class]

    st.markdown(f"### 🧠 Predicted Class: **{predicted_label}**")
    st.markdown(f"#### 🔍 Confidence: **{confidence:.2f}%**")

    # Speak the result
    speak_js(f"Caution. {predicted_label}")
