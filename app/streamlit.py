import streamlit as st
import requests

# API endpoint
API_URL = "http://127.0.0.1:8000/process-audio"

# Streamlit UI
def app():
    st.title("Speech Transcription and Diarization")
    st.markdown("Upload an audio file to process it for transcription and speaker diarization.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

    if uploaded_file:
        with st.spinner("Uploading and processing your audio file..."):
            try:
                # Send the file to the FastAPI server
                response = requests.post(API_URL, files={"file": uploaded_file})

                if response.status_code == 200:
                    result = response.json()

                    # Display transcription
                    st.success("Transcription and diarization completed!")
                    st.markdown("### Transcription Segments:")
                    for segment in result["transcription"]:
                        st.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")

                    # Display diarization
                    st.markdown("### Speaker Diarization:")
                    for entry in result["diarization"]:
                        st.write(entry)
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error occurred')}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()
