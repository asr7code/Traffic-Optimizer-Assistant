import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import cv2
import os

# Constants
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_model()

# Load class names from signnames.csv
@st.cache_data
def load_class_names():
    df = pd.read_csv("signnames.csv")
    return df

sign_df = load_class_names()

st.set_page_config(page_title="GTSRB Traffic Sign Classifier")
st.title("🚦 GTSRB Traffic Sign Classifier")
st.write("Upload an image of a German traffic sign and the model will predict its class!")

uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    pred_class_id = np.argmax(predictions, axis=1)[0]

    # Get the class name
    class_name = sign_df.loc[sign_df['ClassId'] == pred_class_id, 'SignName'].values[0]

    st.success(f"🧠 Predicted: **{class_name}** (Class ID: {pred_class_id})")

    # Optionally: Show prediction confidence
    confidence = np.max(predictions) * 100
    st.info(f"📊 Confidence: {confidence:.2f}%")
