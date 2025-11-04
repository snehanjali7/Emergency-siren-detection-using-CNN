import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("cnn_features_model.h5")
scaler = joblib.load("cnn_scaler.pkl")

# Feature extraction function
def extract_features(signal, sr):
    features = []

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=signal).T, axis=0)
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr).T, axis=0)
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr).T, axis=0)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr).T, axis=0)

    stft = np.abs(librosa.stft(y=signal))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y=signal), sr=sr).T, axis=0)

    features.extend([zcr, rms, spec_cent, spec_bw, rolloff, chroma, mfcc, mel, contrast, tonnetz])
    return np.hstack(features)

# Preprocessing: scaling, padding, reshaping
def preprocess_audio(file):
    signal, sr = librosa.load(file, res_type="kaiser_fast")
    features = extract_features(signal, sr)

    # Scale
    features_scaled = scaler.transform([features])

    # Pad to 192 features
    padded = np.pad(features_scaled[0], (0, 192 - features_scaled.shape[1]), mode="constant")

    # Reshape for CNN
    reshaped = padded.reshape(1, 16, 12, 1)
    return reshaped

# Streamlit UI
st.title("🚨 Emergency Sound Classifier")
st.write("Upload a `.wav` file to classify as **Emergency** or **Non-Emergency**")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    input_tensor = preprocess_audio(uploaded_file)
    prediction = model.predict(input_tensor)
    predicted_class = np.argmax(prediction)

    labels = {
        0: "Non-Emergency",
        1: "Emergency"
    }

    st.markdown(f"### 🧠 Prediction: **{labels[predicted_class]}**")
    st.write(f"🔍 Model confidence: {prediction[0][predicted_class]:.2f}")
    