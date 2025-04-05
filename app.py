import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import cv2
import tempfile

# Set page config
st.set_page_config(page_title="GTSRB Traffic Sign Classifier", layout="centered")

# Load the trained model
@st.cache_resource
def load_cnn_model():
    return load_model("best_model.h5")

# Load the label binarizer
@st.cache_resource
def load_label_binarizer():
    with open("label_binarizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_cnn_model()
label_binarizer = load_label_binarizer()
class_names = label_binarizer.classes_

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Read with OpenCV (like Colab)
    img = cv2.imread(tmp_path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (64, 64))  # Resize to match training input
    img = img.astype("float32") / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

# Streamlit UI
st.title("🚦 German Traffic Sign Classifier")
st.write("Upload an image of a German traffic sign and I'll try to predict it!")

uploaded_file = st.file_uploader("📤 Upload Traffic Sign Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess and predict
    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    predicted_label = class_names[predicted_class]
    confidence = prediction[0][predicted_class] * 100

    # Output result
    st.success(f"✅ **Predicted Traffic Sign Class:** {predicted_label}")
    st.info(f"🔍 Confidence: {confidence:.2f}%")
