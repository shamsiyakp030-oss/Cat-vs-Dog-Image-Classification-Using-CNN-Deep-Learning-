import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: radial-gradient(circle at 12% 0%, rgba(251,146,60,0.12), transparent 45%),
                radial-gradient(circle at 88% 8%, rgba(45,212,191,0.10), transparent 45%),
                linear-gradient(160deg, #1a1410 0%, #15191c 45%, #14110d 100%);
}

.main {
    background: transparent;
}

.title {
    text-align:center;
    font-size:55px;
    font-weight:800;
    margin-top:-40px;
    background: linear-gradient(90deg, #fb923c, #facc15 50%, #2dd4bf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.5px;
}

.subtitle{
    text-align:center;
    color:#a8a29e;
    font-size:18px;
    margin-bottom:30px;
    letter-spacing: 0.3px;
}

.metric-card{
    background: linear-gradient(160deg, #131c2e, #0f1726);
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:#e2e8f0;
    border:1px solid #1e293b;
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
}

.metric-card h3{
    color:#fb923c;
    font-size:15px;
    font-weight:600;
    letter-spacing:0.08em;
    text-transform:uppercase;
    margin-bottom:6px;
}

.metric-card h2{
    color:#f1f5f9;
    font-weight:700;
}

.prediction{
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
    border:1px solid rgba(255,255,255,0.08);
    box-shadow: 0 6px 24px rgba(0,0,0,0.4);
}

.cat{
    background: linear-gradient(135deg,#0f766e,#065f46);
    color:white;
}

.dog{
    background: linear-gradient(135deg,#1e40af,#1d4ed8);
    color:white;
}

.unknown{
    background: linear-gradient(135deg,#9a3412,#7c2d12);
    color:white;
}

.stButton>button{
    width:100%;
    background: linear-gradient(135deg,#fb923c,#2dd4bf);
    color:white;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    padding:12px;
    transition: all 0.2s ease;
}

.stButton>button:hover{
    background: linear-gradient(135deg,#ea7c1f,#14b8a6);
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(45,212,191,0.35);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_cnn():
    import os

    st.write("Current Directory:", os.getcwd())
    st.write("Files:", os.listdir())

    model_path = "cat_dog_model.keras"

    st.write("Model exists:", os.path.exists(model_path))

    return load_model(model_path)
# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown(
    '<div class="title">🐾 Cat vs Dog CNN Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning Image Classification using Convolutional Neural Networks</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
    <h3>Model</h3>
    <h2>CNN</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
    <h3>Input Size</h3>
    <h2>128 × 128</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
    <h3>Classes</h3>
    <h2>Cat / Dog</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        if st.button("🔍 Predict Image"):

            img = image.resize((128,128))

            img_array = img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            score = model.predict(
                img_array,
                verbose=0
            )[0][0]

            confidence = float(score)

            if score <= 0.53:
                label = "🐱 CAT"
                css = "cat"
                conf = (1-score)*100

            elif score > 0.54:
                label = "🐶 DOG"
                css = "dog"
                conf = score*100

            else:
                label = "❓ UNKNOWN OBJECT"
                css = "unknown"
                conf = 50

            st.markdown(
                f"""
                <div class="prediction {css}">
                {label}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Prediction Confidence")

            st.progress(min(int(conf),100))

            st.metric(
                "Confidence Score",
                f"{conf:.2f}%"
            )

            st.metric(
                "Raw Prediction Score",
                f"{confidence:.4f}"
            )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <center>
    <h4 style='color:#94a3b8;'>
    CNN Deep Learning Project • Cat vs Dog Classification
    </h4>
    </center>
    """,
    unsafe_allow_html=True
)
