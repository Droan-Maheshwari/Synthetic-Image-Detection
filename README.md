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

How to Run Locally
Clone the repository:
git clone [https://github.com/your-username/your-repo-name.git]([https://github.com/your-username/your-repo-name.git](https://github.com/Droan-Maheshwari/Synthetic-Image-Detection.git))

cd your-repo-name

Create and activate a virtual environment:
python -m venv myenv
source myenv/bin/activate  # or .\myenv\Scripts\activate on Windows

Install the required packages:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py
