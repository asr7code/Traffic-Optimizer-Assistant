import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import cv2
import tempfile

# Set page config (must be first)
st.set_page_config(page_title="GTSRB Traffic Sign Classifier", layout="centered")

# Load the trained model
@st.cache_resource
def load_cnn_model():
    return load_model("best_model.h5")

model = load_cnn_model()

# Load the label binarizer to get class names
@st.cache_resource
def load_label_binarizer():
    with open("label_binarizer.pkl", "rb") as f:
        return pickle.load(f)

label_binarizer = load_label_binarizer()
class_names = label_binarizer.classes_

def preprocess_pil(pil_img):
    """
    Preprocess a PIL image:
      - Resize to (64, 64)
      - Convert to a NumPy array and normalize pixel values
      - Add a batch dimension
    """
    pil_img = pil_img.resize((64, 64))
    img_array = np.array(pil_img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

def tta_predict(uploaded_file):
    """
    Applies Test-Time Augmentation (TTA) by rotating the image by several angles,
    predicting each variation, and averaging the predictions.
    """
    # Open the uploaded file as a PIL image
    pil_img = Image.open(uploaded_file).convert("RGB")
    
    # Define a few rotation angles (in degrees)
    angles = [-10, 0, 10]
    preds = []
    
    # Loop over each angle, rotate and predict
    for angle in angles:
        rotated = pil_img.rotate(angle)
        processed = preprocess_pil(rotated)
        preds.append(model.predict(processed))
    
    # Average predictions over all augmented versions
    avg_pred = np.mean(preds, axis=0)
    return avg_pred

st.title("🚦 German Traffic Sign Classifier")
st.write("Upload an image of a German traffic sign to classify it:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Use Test-Time Augmentation for a more robust prediction
    prediction = tta_predict(uploaded_file)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    
    st.markdown(f"### 🧠 Predicted Class: **{class_names[predicted_class]}**")
    st.markdown(f"#### 🔍 Confidence: **{confidence:.2f}%**")
