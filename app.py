import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from gtts import gTTS
import os
import tempfile

# Load the model
model = tf.keras.models.load_model("best_model.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

st.set_page_config(page_title="Traffic Sign Assistant", layout="centered")
st.title("🚦 Traffic Sign Recognition & Voice Alert")
st.write("Upload a traffic sign image to get a prediction and voice alert.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((64, 64))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    label = labels[class_id]

    st.success(f"🧠 Prediction: **{label}** with {confidence * 100:.2f}% confidence")

    # Voice Alert
    tts = gTTS(text=f"Attention. {label}", lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")
