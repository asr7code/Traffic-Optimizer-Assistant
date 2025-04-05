import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

# Load trained model
model = tf.keras.models.load_model("best_model.h5")

# Load label binarizer
@st.cache_resource
def load_label_binarizer():
    with open("label_binarizer.pkl", "rb") as f:
        return pickle.load(f)

label_binarizer = load_label_binarizer()
class_names = label_binarizer.classes_

# Preprocessing function
import cv2
import tempfile

def preprocess_image(uploaded_file):
    # Save uploaded file to temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Read using OpenCV to match Colab behavior
    img = cv2.imread(tmp_path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# Streamlit UI
st.title("🚦 GTSRB Traffic Sign Classifier")
st.write("Upload an image of a traffic sign:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    class_label = class_names[predicted_class]

    st.markdown(f"### ✅ Predicted Class: **{class_label}**")
