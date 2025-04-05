import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load class names from labels.txt
@st.cache_data
def load_class_names():
    with open("labels.txt", "r") as file:
        labels = [line.strip() for line in file.readlines()]
    return labels

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5")
    return model

# Preprocess uploaded image
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((30, 30))
    image_np = np.array(image)
    image_np = image_np / 255.0  # Normalize
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

# Main Streamlit App
def main():
    st.title("Traffic Sign Classifier 🚦")
    st.write("Upload a traffic sign image and I'll tell you what it is!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("Classifying...")
        processed_image = preprocess_image(image)

        model = load_model()
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions)
        class_names = load_class_names()
        predicted_label = class_names[predicted_class]

        st.success(f"Prediction: **{predicted_label}**")

if __name__ == "__main__":
    main()
