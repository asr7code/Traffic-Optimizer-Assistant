import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("best_model.h5")

# Traffic light class labels (replace with yours if different)
class_names = ['Red', 'Yellow', 'Green']  # or all 43 if you have 43 classes

# Expected input size from model summary
expected_size = (62, 62)

# App UI
st.title("🚦 Traffic Light Classifier")
st.write("Upload an image of a traffic light to identify its signal color.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Resize to model's expected input size
        image = image.resize(expected_size)
        image_array = np.array(image) / 255.0  # normalize
        image_array = np.expand_dims(image_array, axis=0)  # add batch dimension

        # Predict
        prediction = model.predict(image_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Output
        st.success(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: {np.max(prediction)*100:.2f}%")

    except Exception as e:
        st.error("⚠️ Error during prediction.")
        st.text(f"Details: {e}")
