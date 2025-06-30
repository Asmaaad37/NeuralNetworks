import streamlit as st
import cv2
import tempfile
import numpy as np
import os
from PIL import Image
import gc
import tensorflow
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # or ConvNeXtBase, depending on your model



# Set page config
st.set_page_config(page_title="DeepFake Detector", layout="centered", initial_sidebar_state="collapsed")

# Minimal dark mode style
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, .stApp {
        background-color: #0e0e0e;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    .block-container {
        padding: 2rem 2rem;
    }

    h1, h2, h3, .stMarkdown, .stFileUploader, .stVideo, .stImage, .stButton, .stProgress, .score-box {
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    h1, h2, h3 {
        color: #fefefe;
        font-weight: 600;
        border-left: 4px solid #00e5ff;
        padding-left: 10px;
    }

    .stButton>button {
        background-color: #00e5ff;
        color: black;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        border: none;
        transition: 0.3s;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .stButton>button:hover {
        background-color: #00bcd4;
        transform: scale(1.05);
    }

    .stFileUploader {
        background-color: #1e1e1e;
        padding: 1em;
        border-radius: 12px;
        border: 1px dashed #444;
    }

    .stVideo, .stImage > img {
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .stProgress > div > div > div > div {
        background-color: #00e5ff;
    }

    .score-box {
        background-color: #1e1e1e;
        padding: 1em;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #00e5ff;
        font-size: 1.1rem;
    }

</style>
""", unsafe_allow_html=True)



# Logo
# st.markdown("""
# <a href="https://www.intel.com/content/www/us/en/research/fakecatcher.html" target="_blank">
#     <img src="https://cdn-icons-png.flaticon.com/512/10471/10471465.png" width="100">
# </a>
# """, unsafe_allow_html=True)

# App title & subtitle
st.title("🕵️‍♂️ DeepFake Detector")
st.markdown("Upload a video and detect deepfakes using an AI-based model — powered by Xception!")

st.markdown("""
<div class='animated-desc'>
Analyze your video content for potential deepfake alterations using cutting-edge frame-by-frame AI detection.
</div>
""", unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("📤 Upload a video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
@st.cache_resource
def load_model():
    weights_path = "Xception_ft.weights.h5"
    base_model = Xception(weights=None, include_top=False, input_shape=(299, 299, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights(weights_path)
    return model

model = load_model()


# Real detector function
def real_fake_detector(frame: np.ndarray) -> float:
    resized = tf.image.resize(frame, (299, 299)) / 255.0
    resized = tf.expand_dims(resized, axis=0)
    prediction = model.predict(resized, verbose=0)[0][0]
    return float(prediction)

# Main logic
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Show uploaded video
    st.video(uploaded_file)

    st.info("🧠 Extracting frames and analyzing with local model...")
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = []
    fake_scores = []
    progress = st.progress(0)

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if i % 20 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            selected_frames.append(frame_rgb)
            score = real_fake_detector(frame_rgb)
            fake_scores.append(score)

        progress.progress(min((i + 1) / frame_count, 1.0))

    cap.release()

    # Display a few sample frames
    st.subheader("📷 Sample Extracted Frames")
    cols = st.columns(min(len(selected_frames), 5))
    for idx, img in enumerate(selected_frames[:5]):
        with cols[idx]:
            st.image(Image.fromarray(img), use_column_width=True)

    avg_score = np.mean(fake_scores)
    st.markdown("---")
    st.subheader("📊 DeepFake Analysis Result")

    with st.container():
        st.markdown(f"""
        <div class='score-box'>
            <strong>Average Fake Score:</strong> {avg_score:.2f}<br>
            <strong>Confidence Level:</strong> {'⚠ High' if avg_score > 0.75 else '✅ Moderate'}
        </div>
    """, unsafe_allow_html=True)

    if avg_score > 0.5:
        st.error("🔍 The video is likely **DeepFake**.")
    else:
        st.success("🎉 The video appears **Authentic**.")
