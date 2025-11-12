# Synthetic-Image-Detection
A deep learning model built with TensorFlow/Keras and deployed as a Streamlit app to classify images as either real photographs or AI-generated.

This project is an interactive web application built with Streamlit that uses a deep learning model to predict whether an image is a real photograph or AI-generated.
The model is a Convolutional Neural Network (CNN) built with TensorFlow and Keras, utilizing transfer learning (VGG16) to achieve high accuracy.

Features
Image Upload: Users can upload JPG, PNG, or JPEG files for analysis.
Instant Prediction: The model provides a real-time prediction (REAL vs. AI-GENERATED) and a confidence score.
Feedback Loop: A simple "Yes/No" feedback system saves images that the model gets right or wrong, allowing for easy data collection to improve the model over time.

Tech Stack
Python
Streamlit (for the web interface)
TensorFlow / Keras (for the deep learning model)
Pillow (for image processing)
NumPy (for numerical operations)

Folder Structure
project/
│
├── app.py                  # Streamlit web interface
│
├── models/
│   └── best_real_vs_ai_detector.h5   # Model file (auto-downloaded from Hugging Face)
│
├── feedback/               # Saved feedback images from the Streamlit app
│   ├── correct_real/
│   ├── correct_ai/
│   ├── incorrect_real/
│   └── incorrect_ai/
│
├── requirements.txt        # Dependencies for app and scripts
└── README.md               # Documentation


💻 Usage
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the Streamlit app
streamlit run app.py


The first run will automatically download the model from Hugging Face into the models/ folder.

Then open the app in your browser:
http://localhost:8501


Upload an image, click Analyze, and the app will tell you if it’s Real or AI-Generated — with confidence percentage.

You can also provide feedback (Yes/No) to help future retraining.

For direct use - https://synthetic-image-detection-g6aeud7as2gmhmhon9qby3.streamlit.app/


Install the required packages:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py
