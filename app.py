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
def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((64, 64))  # Ensure same size as training
    img_array = np.array(img).astype("float32") / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

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
