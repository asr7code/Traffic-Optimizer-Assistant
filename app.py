import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import cv2
import tempfile

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

def preprocess_image(uploaded_file):
    """
    Preprocess the uploaded image to match training conditions:
    - Uses OpenCV to mimic training (BGR -> RGB conversion).
    - Resizes to 64x64.
    - Normalizes pixel values.
    - Adds batch dimension.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    img = cv2.imread(tmp_path)  # BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

st.title("🚦 German Traffic Sign Classifier")
st.write("Upload an image of a German traffic sign to classify it:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

    st.markdown(f"### 🧠 Predicted Class: **{class_names[predicted_class]}**")
    st.markdown(f"#### 🔍 Confidence: **{confidence:.2f}%**")
    
    # Display Top-3 Predictions for further diagnosis
    top3_indices = np.argsort(prediction[0])[-3:][::-1]
    st.markdown("#### Top 3 Predictions:")
    for idx in top3_indices:
        st.write(f"**{class_names[idx]}**: {prediction[0][idx]*100:.2f}% confidence")
