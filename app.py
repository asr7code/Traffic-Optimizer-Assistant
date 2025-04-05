import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import pickle

# 🚦 Set page config (MUST be first Streamlit command)
st.set_page_config(page_title="Traffic Sign Voice Alert", layout="centered")

# 📌 Manual mapping from class number to label
CLASS_LABELS = {
    0: "Speed limit 20 km per hour",
    1: "Speed limit 30 km per hour",
    2: "Speed limit 50 km per hour",
    3: "Speed limit 60 km per hour",
    4: "Speed limit 70 km per hour",
    5: "Speed limit 80 km per hour",
    6: "End of speed limit 80 km per hour",
    7: "Speed limit 100 km per hour",
    8: "Speed limit 120 km per hour",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right of way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice or snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons"
}

# 📦 Load CNN model
@st.cache_resource
def load_cnn_model():
    return load_model("best_model.h5")

model = load_cnn_model()

# 📂 Preprocess uploaded image
def preprocess_image(image_data):
    image = Image.open(image_data).convert("RGB")
    image_np = np.array(image)
    image_bgr = image_np[..., ::-1]
    image_bgr = cv2.resize(image_bgr, (64, 64))
    image_bgr = image_bgr.astype("float32") / 255.0
    return np.expand_dims(image_bgr, axis=0)

# 🗣️ Inject JavaScript for browser voice
def auto_speak_js(label_text):
    safe_label = label_text.replace('"', '\\"')
    js_code = f"""
        <script>
        var msg = new SpeechSynthesisUtterance("Caution! {safe_label}");
        msg.lang = 'en-US';
        msg.rate = 0.9;
        speechSynthesis.speak(msg);
        </script>
    """
    st.components.v1.html(js_code)

# 🌟 UI
st.title("🚘 Traffic Sign Voice Alert System")
st.write("Upload an image of a traffic sign to hear a voice alert for drivers.")

uploaded_file = st.file_uploader("Upload traffic sign image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(uploaded_file)
    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index] * 100

    predicted_label = CLASS_LABELS.get(predicted_index, f"Class {predicted_index}")

    st.markdown(f"### 🧠 Predicted Sign: **{predicted_label}**")
    st.markdown(f"#### 🔍 Confidence: **{confidence:.2f}%**")

    auto_speak_js(predicted_label)
