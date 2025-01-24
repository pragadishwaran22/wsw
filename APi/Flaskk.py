
import streamlit as st
import whisper
from pydub import AudioSegment
from pydub.utils import which
import os
from pyannote.audio import Pipeline
import datetime
from concurrent.futures import ThreadPoolExecutor

# Set FFmpeg path for pydub
AudioSegment.converter = which("ffmpeg")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []  # Stores history entries

# Function to save results to history
def save_to_history(name, transcript_type, transcription, diarization):
    st.session_state.history.append({
        "name": name,
        "type": transcript_type,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "transcription": transcription,
        "diarization": diarization
    })

# Function to preprocess audio
def preprocess_audio(input_file, output_file="output.wav", target_sample_rate=16000):
    try:
        audio = AudioSegment.from_file(input_file)
        if input_file.endswith(".wav"):
            with open(input_file, "rb") as f:
                header = f.read(44)
                sample_rate = int.from_bytes(header[24:28], byteorder="little")
                if sample_rate == target_sample_rate:
                    return input_file

        audio = audio.set_frame_rate(target_sample_rate).set_channels(1)
        audio.export(output_file, format="wav")
        return output_file

    except Exception as e:
        raise RuntimeError(f"Error during audio preprocessing: {e}")

# Whisper transcription with timestamps
def transcribe_audio_with_segments(audio_file, model_type="medium"):
    try:
        model = whisper.load_model(model_type)
        result = model.transcribe(audio_file, word_timestamps=True)
        return result["segments"]

    except FileNotFoundError:
        raise RuntimeError("FFmpeg is not found. Ensure it is installed and accessible.")
    except Exception as e:
        raise RuntimeError(f"Error during transcription: {e}")

# Speaker diarization using Pyannote
def diarize_audio(audio_file):
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
        diarization = pipeline({"uri": "audio", "audio": audio_file})
        
        diarization_results = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            diarization_results.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })
        
        return diarization_results

    except Exception as e:
        raise RuntimeError(f"Error during diarization: {e}")

# Align transcription with diarization
def align_transcription_with_diarization(transcription_segments, diarization_segments):
    output = []
    used_transcription_indices = set()

    for diarization_segment in diarization_segments:
        speaker = diarization_segment["speaker"]
        start = diarization_segment["start"]
        end = diarization_segment["end"]

        matching_transcription_segments = [
            (i, segment) for i, segment in enumerate(transcription_segments)
            if i not in used_transcription_indices and segment["start"] < end and segment["end"] > start
        ]

        speaker_text = []
        for i, segment in matching_transcription_segments:
            overlap_start = max(start, segment["start"])
            overlap_end = min(end, segment["end"])
            if overlap_end > overlap_start:
                speaker_text.append(segment["text"])
                used_transcription_indices.add(i)

        output.append(f"Speaker {speaker} [{start:.2f}s - {end:.2f}s]: {' '.join(speaker_text)}")

    return output

# Process a single file
def process_file(uploaded_file):
    try:
        temp_input_file = f"temp_{uploaded_file.name}"
        with open(temp_input_file, "wb") as f:
            f.write(uploaded_file.read())

        preprocessed_audio = preprocess_audio(temp_input_file)
        transcription_segments = transcribe_audio_with_segments(preprocessed_audio)
        diarization_segments = diarize_audio(preprocessed_audio)
        aligned_output = align_transcription_with_diarization(transcription_segments, diarization_segments)

        save_to_history(uploaded_file.name, "Recorded", transcription_segments, aligned_output)

        os.remove(temp_input_file)
        if os.path.exists("output.wav"):
            os.remove("output.wav")

        return uploaded_file.name, aligned_output

    except Exception as e:
        return uploaded_file.name, f"Error: {e}"

# Streamlit app
def app():
    st.title("Batch Speech Diarization and Transcription")
    st.markdown("Upload multiple audio files (MP3, WAV, M4A) for transcription and speaker diarization.")

    uploaded_files = st.file_uploader("Choose audio files", type=["mp3", "wav", "m4a"], accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Processing audio files..."):
            results = []
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_file, file) for file in uploaded_files]
                for future in futures:
                    results.append(future.result())

        st.success("Processing complete!")
        for file_name, result in results:
            st.markdown(f"### Results for {file_name}:")
            if isinstance(result, list):
                for entry in result:
                    st.write(entry)
            else:
                st.error(result)

# Run the app
if __name__ == "__main__":
    app()