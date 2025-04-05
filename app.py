import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Traffic Sign Classifier", layout="centered")

# Title and description
st.title("🚦 Traffic Sign Classifier")
st.write("Upload an image of a traffic sign, and the model will predict what it is.")

# Load class names from signnames.csv
@st.cache_data
def load_class_names():
    df = pd.read_csv("signnames.csv")
    return dict(zip(df["ClassId"], df["SignName"]))

class_names = load_class_names()

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload a traffic sign image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((30, 30))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 30, 30, 3)

    # Predict
    prediction = model.predict(img_array)
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # Display result
    st.success(f"🧠 Predicted Sign: **{class_names[class_id]}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")
