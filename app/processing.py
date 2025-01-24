from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from pydub import AudioSegment
from pydub.utils import which
import whisper
from pyannote.audio import Pipeline
import os
import datetime
import uvicorn
import tempfile

# Initialize FastAPI apppip install fastapi uvicorn torch openai-whisper pyannote.audio pydub soundfile python-decouple
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Speech Processing API!"}

# Set FFmpeg path for pydub
AudioSegment.converter = which("ffmpeg")

# Initialize Pyannote pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", 
    use_auth_token="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"
)

# Load Whisper model
model = whisper.load_model("medium")

class ProcessingResult(BaseModel):
    transcription: list
    diarization: list

# Helper functions
def preprocess_audio(input_file: str, output_file: str = "output.wav", target_sample_rate: int = 16000):
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(target_sample_rate).set_channels(1)
    audio.export(output_file, format="wav")
    return output_file

def transcribe_audio_with_segments(audio_file: str):
    result = model.transcribe(audio_file, word_timestamps=True)
    return result["segments"]

def diarize_audio(audio_file: str):
    diarization = pipeline({"uri": "audio", "audio": audio_file})
    diarization_results = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        diarization_results.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker
        })
    return diarization_results

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

# FastAPI routes
@app.post("/process-audio", response_model=ProcessingResult)
async def process_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file to temporary file
        temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]).name
        with open(temp_input_file, "wb") as f:
            f.write(file.file.read())

        # Preprocess audio
        preprocessed_audio = preprocess_audio(temp_input_file)

        # Transcription
        transcription_segments = transcribe_audio_with_segments(preprocessed_audio)

        # Diarization
        diarization_segments = diarize_audio(preprocessed_audio)

        # Align transcription with diarization
        aligned_output = align_transcription_with_diarization(transcription_segments, diarization_segments)

        return ProcessingResult(transcription=transcription_segments, diarization=aligned_output)

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Clean up temporary files
        if os.path.exists(temp_input_file):
            os.remove(temp_input_file)
        if os.path.exists("output.wav"):
            os.remove("output.wav")

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
