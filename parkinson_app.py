import streamlit as st
import librosa
import numpy as np
import pandas as pd
from keras.models import load_model
import soundfile as sf

# Load the trained model
model = load_model("best_cnn_bilstm_model.keras")

# Define the mapping for class labels
class_labels = {
    0: "Healthy",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

# Extract features matching the dataset
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)  # Resample to 22050 Hz
    
    features = {
        "MDVP:Fo(Hz)": np.mean(librosa.yin(y, fmin=50, fmax=500, sr=sr)),
        "MDVP:Fhi(Hz)": np.max(librosa.yin(y, fmin=50, fmax=500, sr=sr)),
        "MDVP:Flo(Hz)": np.min(librosa.yin(y, fmin=50, fmax=500, sr=sr)),
        "MDVP:Jitter(%)": np.std(librosa.zero_crossings(y)) * 100,
        "MDVP:Jitter(Abs)": np.mean(np.abs(np.diff(y))),
        "MDVP:RAP": np.mean(librosa.feature.rms(y=y)),
        "MDVP:PPQ": np.std(librosa.feature.rms(y=y)),
        "Jitter:DDP": np.var(librosa.feature.rms(y=y)),
        "MDVP:Shimmer": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "MDVP:Shimmer(dB)": np.std(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "Shimmer:APQ3": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "Shimmer:APQ5": np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "MDVP:APQ": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "Shimmer:DDA": np.std(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "NHR": np.mean(librosa.feature.spectral_flatness(y=y)),
        "HNR": np.mean(librosa.feature.rms(y=y)),
        "RPDE": np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
        "DFA": np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
        "spread1": np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)),
        "spread2": np.std(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)),
        "D2": np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        "PPE": np.std(librosa.feature.mfcc(y=y, sr=sr)),
    }
    
    df = pd.DataFrame([features])
    return df

# Streamlit UI
st.title("Parkinson's Disease Multi-Class Prediction")
st.write("Upload a voice recording (WAV format) to predict the Parkinson’s stage.")

audio_file = st.file_uploader("Upload Audio File", type=["wav"])

if audio_file is not None:
    # Save uploaded file to a temp file
    with open("temp.wav", "wb") as f:
        f.write(audio_file.read())

    st.audio("temp.wav")

    # Extract features
    df = extract_features("temp.wav")

    # Display extracted features
    st.subheader("Extracted Features")
    st.dataframe(df.round(4), use_container_width=True)

    # Reshape input for prediction
    input_data = np.expand_dims(df.values, axis=2)  # Add channel dimension

    # Predict
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]

    st.subheader("Prediction Result")
    st.success(f"Predicted Class: {predicted_class} - **{predicted_label}**")
