import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("best_model.h5")

# Define class names (change this according to your training labels)
class_names = ['Green', 'Red', 'Yellow']

# Streamlit App
st.title("🚦 Traffic Light Classifier")
st.write("Upload an image of a traffic light to identify its color.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing: Resize, normalize, and expand dimensions
    image = image.resize((128, 128))  # change if your model was trained on different size
    image_array = np.array(image) / 255.0  # normalize
    image_array = np.expand_dims(image_array, axis=0)  # shape: (1, 128, 128, 3)

    # Predict
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Display result
    st.success(f"**Predicted Class: {predicted_class}**")
    st.write(f"Prediction Confidence: {np.max(prediction)*100:.2f}%")
