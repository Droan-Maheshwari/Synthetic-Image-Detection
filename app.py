import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import time

# --- Configuration ---
MODEL_PATH = 'models/best_real_vs_ai_detector.h5'
IMG_WIDTH, IMG_HEIGHT = 150, 150
FEEDBACK_DIR = "feedback"

# --- Create Feedback Directory ---
if not os.path.exists(FEEDBACK_DIR):
    os.makedirs(os.path.join(FEEDBACK_DIR, "correct_real"))
    os.makedirs(os.path.join(FEEDBACK_DIR, "correct_ai"))
    os.makedirs(os.path.join(FEEDBACK_DIR, "incorrect_real"))
    os.makedirs(os.path.join(FEEDBACK_DIR, "incorrect_ai"))

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure the 'best_real_vs_ai_detector.h5' file is in the 'models' directory.")
        return None

model = load_model()

# --- Helper Functions ---

def preprocess_image(pil_image):
    """
    Preprocesses the PIL image to match the model's input requirements.
    Returns a numpy array.
    """
    img = ImageOps.fit(pil_image, (IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array_scaled = img_array / 255.0  # Rescale for prediction
    return img_array_scaled

def save_feedback(pil_image, prediction, is_correct):
    """Saves the image to the feedback directory."""
    try:
        timestamp = int(time.time())
        if is_correct:
            if prediction == "REAL":
                folder = "correct_real"
            else:
                folder = "correct_ai"
        else:
            if prediction == "REAL":
                folder = "incorrect_real" # Model said REAL, but it was (presumably) AI
            else:
                folder = "incorrect_ai"   # Model said AI, but it was (presumably) REAL
                
        save_path = os.path.join(FEEDBACK_DIR, folder, f"{timestamp}.png")
        pil_image.save(save_path)
        st.toast(f"Feedback saved! Thank you!", icon="👍")
    except Exception as e:
        st.error(f"Error saving feedback: {e}")


# --- Streamlit UI ---

st.set_page_config(page_title="AI vs. Real Image Detector", layout="wide")
st.title("🤖 AI vs. Real Image Detector")
st.write("Upload an image, and the model will predict if it's REAL or AI-GENERATED.")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Upload an Image")
    
    # --- Input Field ---
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

    image = None
    image_source_name = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_source_name = uploaded_file.name

    # --- Analysis and Display ---
    if image is not None:
        # Display the image
        st.image(image, caption=f'Uploaded: {image_source_name}', use_container_width=True)
        
        # Analyze button
        if st.button('Analyze Image', key="analyze_button"):
            if model is None:
                st.error("Model is not loaded. Cannot perform analysis.")
            else:
                with st.spinner('Analyzing...'):
                    # Preprocess image
                    img_array_scaled = preprocess_image(image)
                    
                    # Make prediction
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
                    
                    # Store for feedback
                    st.session_state.last_image = image
                    st.session_state.analyzed = True

# --- Right Column for Results ---
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

        # --- Feedback Loop ---
        st.subheader("Was this prediction correct?")
        fb_col1, fb_col2 = st.columns(2)
        
        with fb_col1:
            if st.button("Yes", key="feedback_yes", use_container_width=True):
                save_feedback(st.session_state.last_image, prediction, is_correct=True)
                # Clear state after feedback
                st.session_state.analyzed = False
                st.rerun()

        with fb_col2:
            if st.button("No", key="feedback_no", use_container_width=True):
                save_feedback(st.session_state.last_image, prediction, is_correct=False)
                # Clear state after feedback
                st.session_state.analyzed = False
                st.rerun()

# --- Logic to clear results when a new image is uploaded ---
if (uploaded_file is not None) and not st.session_state.get('analyze_button', False):
     if 'analyzed' in st.session_state:
         del st.session_state.analyzed
     if 'prediction' in st.session_state:
         del st.session_state.prediction