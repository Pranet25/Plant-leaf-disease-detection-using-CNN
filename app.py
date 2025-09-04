import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# Page config
st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ======== Background & Glass UI ========
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/jpg;base64,{encoded}") no-repeat center center fixed;
        background-size: cover;
    }}
    [data-testid="stAppViewContainer"] .main {{
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        max-width: 1000px;
        margin: auto;
    }}
    .main > div {{
        max-width: 700px;
        margin: auto;
    }}
    h1, h2, h3, p, label {{
        color: #ffffff !important;
        text-shadow: 0px 2px 5px rgba(0,0,0,0.6);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

add_bg_from_local("background.jpg")

# ======== Load disease model ========
@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model("plant_cnn_mobilenet.h5")

disease_model = load_disease_model()

# ======== Class names & remedies ========
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

remedies = {
    "Potato___Early_blight": "Apply fungicides (chlorothalonil, mancozeb), remove infected leaves.",
    "Potato___Late_blight": "Use resistant varieties, apply fungicides (metalaxyl, chlorothalonil).",
    "Tomato_Early_blight": "Use fungicides, prune lower leaves, ensure crop rotation.",
    "Tomato_Late_blight": "Remove infected plants, avoid overhead irrigation, use fungicides.",
    "Tomato_Bacterial_spot": "Remove infected leaves, use copper sprays.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Remove infected plants, control whiteflies.",
    "Tomato__Tomato_mosaic_virus": "Disinfect tools, remove infected plants.",
    "Tomato_Leaf_Mold": "Improve air circulation, apply fungicides.",
    "Tomato_Septoria_leaf_spot": "Use drip irrigation, apply fungicides.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Apply miticides, encourage natural predators.",
    "Tomato__Target_Spot": "Ensure good airflow, use fungicides.",
    "Pepper__bell___Bacterial_spot": "Rotate crops, use copper-based fungicide.",
}

# ======== UI ========
st.title("🌿 Potato, Tomato & Bell Pepper Leaf Disease Detection")
st.write("Upload a leaf image to detect its health status and get remedial measures.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess for model
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Disease prediction
    predictions = disease_model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.subheader("Prediction Result")
    st.success(f"**{predicted_class}** with **{confidence:.2f}%** confidence")

    st.subheader("🌱 Recommended Action")
    st.info(remedies.get(predicted_class, "Plant looks healthy."))
