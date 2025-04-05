import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("best_model.h5")

# 43 class names (0–42 as integers)
class_names = list(range(43))  # GTSRB uses numeric class IDs

IMG_SIZE = (64, 64)

st.title("🚦 GTSRB Traffic Sign Classifier")
st.write("Upload a traffic sign image to predict the class (0–42).")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to NumPy
        image_np = np.array(image)

        # Convert RGB to BGR (to match OpenCV training)
        image_bgr = image_np[..., ::-1]

        # Resize to 64x64 (same as training)
        image_resized = tf.image.resize(image_bgr, IMG_SIZE).numpy()

        # Normalize
        image_normalized = image_resized.astype("float32") / 255.0

        # Add batch dimension
        input_array = np.expand_dims(image_normalized, axis=0)

        # Predict
        prediction = model.predict(input_array)
        pred_class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        st.success(f"Predicted Class ID: **{pred_class_id}**")
        st.info(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.text(str(e))
