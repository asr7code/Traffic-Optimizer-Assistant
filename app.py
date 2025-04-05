import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import cv2

# Must be the first Streamlit command
st.set_page_config(page_title="GTSRB Traffic Sign Classifier", layout="centered")

# Load the trained model (using caching to speed up subsequent runs)
@st.cache_resource
def load_cnn_model():
    return load_model("best_model.h5")

model = load_cnn_model()

# Load the label binarizer to get class names (using caching)
@st.cache_resource
def load_label_binarizer():
    with open("label_binarizer.pkl", "rb") as f:
        return pickle.load(f)

label_binarizer = load_label_binarizer()
class_names = label_binarizer.classes_

def preprocess_image(image_data):
    """
    Preprocesses the uploaded image to match the training conditions:
    - Reads image using PIL and converts it to RGB.
    - Converts RGB image to BGR (mimicking cv2.imread() used during training).
    - Resizes the image to 64x64.
    - Normalizes pixel values (divides by 255.0).
    - Adds a batch dimension.
    """
    # Open the uploaded image
    image = Image.open(image_data).convert("RGB")
    
    # Convert to numpy array (in RGB)
    image_np = np.array(image)
    
    # Convert RGB to BGR by reversing the last channel
    image_bgr = image_np[..., ::-1]
    
    # Resize image to (64, 64) using OpenCV for consistency with training
    image_bgr = cv2.resize(image_bgr, (64, 64))
    
    # Normalize pixel values to [0, 1]
    image_bgr = image_bgr.astype("float32") / 255.0
    
    # Add batch dimension so that shape becomes (1, 64, 64, 3)
    return np.expand_dims(image_bgr, axis=0)

# Streamlit UI
st.title("🚦 German Traffic Sign Classifier")
st.write("Upload an image of a German traffic sign to classify it:")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image using the function defined above
    processed_image = preprocess_image(uploaded_file)
    
    # Run the model prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    
    # Display prediction result and confidence
    st.markdown(f"### 🧠 Predicted Class: **{class_names[predicted_class]}**")
    st.markdown(f"#### 🔍 Confidence: **{confidence:.2f}%**")
