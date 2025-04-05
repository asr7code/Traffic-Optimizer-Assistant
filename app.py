import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# 🔧 Must be first Streamlit command
st.set_page_config(page_title="Traffic Sign Classifier", layout="centered")

# 🚀 Load model only once using caching
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5")
    return model

# 🧠 Load the model
model = load_model()

# 📂 Load class names
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# 🧼 Preprocess function for input image
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((64, 64))  # resize to model's expected input size
    img_array = np.array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 64, 64, 3)
    return img_array

# 🧠 App Title
st.title("🚦 Traffic Sign Classifier")
st.write("Upload an image of a traffic sign and I’ll predict what it is.")

# 📤 File uploader
uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "jpeg", "png"])

# 🔮 Predict and display results
if uploaded_file is not None: 
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess image and make prediction
    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    st.write(f"### 🧠 Predicted Traffic Sign Class: **{class_names[predicted_class]}**")
