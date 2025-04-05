import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from gtts import gTTS
import os
import time
from PIL import Image
import tempfile
from playsound import playsound

# Load model and labels
model = load_model("best_model.h5")

with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

st.set_page_config(page_title="Traffic Sign Assistant", layout="centered")

st.title("🚦 Traffic Sign Recognition & Voice Assistant")
st.write("Upload a traffic sign image and the system will recognize and speak the traffic rule.")

uploaded_file = st.file_uploader("📤 Upload Traffic Sign Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((64, 64))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = labels[class_index]
    confidence = np.max(prediction)

    st.success(f"🧠 Prediction: **{class_label}** ({confidence * 100:.2f}%)")

    # Voice alert using gTTS
    tts = gTTS(text=f"Attention! {class_label}", lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")
