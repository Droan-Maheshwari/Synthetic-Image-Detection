# app.py (modified to download model from Hugging Face at runtime)
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import time
from huggingface_hub import hf_hub_download, login
from huggingface_hub.utils import EntryNotFoundError

# --- Configuration ---
REPO_ID = "Droan-Maheshwari/Synthetic-Image-Detector"  # e.g. "yourname/real-vs-ai-detector"
FILENAME = "best_real_vs_ai_detector.h5"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, FILENAME)
IMG_WIDTH, IMG_HEIGHT = 150, 150
FEEDBACK_DIR = "feedback"

# --- Create Feedback Directory ---
if not os.path.exists(FEEDBACK_DIR):
    os.makedirs(os.path.join(FEEDBACK_DIR, "correct_real"), exist_ok=True)
    os.makedirs(os.path.join(FEEDBACK_DIR, "correct_ai"), exist_ok=True)
    os.makedirs(os.path.join(FEEDBACK_DIR, "incorrect_real"), exist_ok=True)
    os.makedirs(os.path.join(FEEDBACK_DIR, "incorrect_ai"), exist_ok=True)

# --- Helper: download model from HF if missing ---
def download_model_from_hf(repo_id: str, filename: str, local_path: str):
    """
    Downloads the file `filename` from `repo_id` on Hugging Face to local_path.
    Returns the local_path if successful, otherwise raises.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        # This will download LFS-tracked files as well.
        st.info(f"Downloading model from Hugging Face: {repo_id}/{filename} ...")
        hf_local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        # hf_hub_download returns a path to the cached file; ensure it's copied/linked to our models directory
        if os.path.abspath(hf_local_path) != os.path.abspath(local_path):
            # Copy file into models dir for consistent path usage
            import shutil
            shutil.copyfile(hf_local_path, local_path)
        st.success("Model downloaded successfully.")
        return local_path
    except EntryNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found in repo '{repo_id}'. Please check the repo and filename.")
    except Exception as e:
        raise RuntimeError(f"Error downloading model from Hugging Face: {e}")

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads (and if needed downloads) the pre-trained Keras model."""
    try:
        if not os.path.exists(MODEL_PATH):
            # Try to download from Hugging Face repo
            try:
                download_model_from_hf(REPO_ID, FILENAME, MODEL_PATH)
            except Exception as e:
                # Informative error for the user.
                st.error(f"Could not download model automatically: {e}")
                st.error("If your model repository is private, run 'huggingface-cli login' or set HF_TOKEN in your environment.")
                return None
            
        # Load model with TF
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Helper Functions ---
def preprocess_image(pil_image):
    img = ImageOps.fit(pil_image, (IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array_scaled = img_array / 255.0
    return img_array_scaled

def save_feedback(pil_image, prediction, is_correct):
    try:
        timestamp = int(time.time())
        if is_correct:
            folder = "correct_real" if prediction == "REAL" else "correct_ai"
        else:
            folder = "incorrect_real" if prediction == "REAL" else "incorrect_ai"
        save_path = os.path.join(FEEDBACK_DIR, folder, f"{timestamp}.png")
        pil_image.save(save_path)
        st.toast("Feedback saved! Thank you!", icon="👍")
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

# --- Streamlit UI ---
st.set_page_config(page_title="AI vs. Real Image Detector", layout="wide")
st.title("🤖 AI vs. Real Image Detector")
st.write("Upload an image, and the model will predict if it's REAL or AI-GENERATED.")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

    image = None
    image_source_name = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_source_name = uploaded_file.name

    if image is not None:
        st.image(image, caption=f'Uploaded: {image_source_name}', use_container_width=True)
        if st.button('Analyze Image', key="analyze_button"):
            if model is None:
                st.error("Model not loaded. Please check logs and ensure model is available.")
            else:
                with st.spinner('Analyzing...'):
                    img_array_scaled = preprocess_image(image)
                    score = model.predict(img_array_scaled)[0][0]
                    confidence = max(score, 1 - score) * 100
                    if score > 0.5:
                        prediction = "REAL"
                        st.session_state.prediction = "REAL"
                        st.session_state.confidence = confidence
                    else:
                        prediction = "AI-GENERATED"
                        st.session_state.prediction = "AI-GENERATED"
                        st.session_state.confidence = confidence
                    st.session_state.last_image = image
                    st.session_state.analyzed = True

with col2:
    if st.session_state.get('analyzed', False):
        st.header("Analysis Results")
        prediction = st.session_state.prediction
        confidence = st.session_state.confidence
        if prediction == "REAL":
            st.success(f"**Prediction: REAL** (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"**Prediction: AI-GENERATED** (Confidence: {confidence:.2f}%)")
        st.write("---")
        st.subheader("Was this prediction correct?")
        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            if st.button("Yes", key="feedback_yes", use_container_width=True):
                save_feedback(st.session_state.last_image, prediction, is_correct=True)
                st.session_state.analyzed = False
                st.rerun()

        with fb_col2:
            if st.button("No", key="feedback_no", use_container_width=True):
                save_feedback(st.session_state.last_image, prediction, is_correct=False)
                st.session_state.analyzed = False
                st.rerun()

# Clear results when a new image is uploaded
if (uploaded_file is not None) and not st.session_state.get('analyze_button', False):
    if 'analyzed' in st.session_state:
        del st.session_state.analyzed
    if 'prediction' in st.session_state:
        del st.session_state.prediction
