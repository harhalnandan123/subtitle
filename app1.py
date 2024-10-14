import os
import streamlit as st
from transformers import pipeline
import librosa
import tempfile

# Initialize the ASR pipeline
asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h-lv60")

# Set up the Streamlit app
st.title("Audio Transcription App")
st.write("Upload a `.wav` file, and this app will transcribe the audio content into text.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"])

# Process the uploaded file
if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(uploaded_file.getbuffer())
        temp_audio_path = temp_audio_file.name

    # Load the audio file with librosa and resample to 16kHz
    audio_data, sr = librosa.load(temp_audio_path, sr=16000)
    
    # Transcribe the audio data
    st.write("Transcribing audio...")

    # Perform transcription and save to a temporary text file
    transcription = asr_pipeline(audio_data)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as temp_text_file:
        temp_text_file.write(transcription['text'])
        temp_text_path = temp_text_file.name

    st.success("Transcription completed! You can read the transcription below or download it.")

    # Button to display the transcription
    if st.button("Read Transcription"):
        with open(temp_text_path, "r") as txt_file:
            st.write(txt_file.read())

    # Option to download the transcription file
    with open(temp_text_path, "r") as txt_file:
        st.download_button("Download Transcription", txt_file, file_name="transcription.txt")

    # Cleanup temporary files
    os.remove(temp_audio_path)
    os.remove(temp_text_path)
else:
    st.write("Please upload a `.wav` file to start transcription.")
