
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline


MODEL_NAME = "pyannote/speaker-diarization-community-1"


def diarize_audio(audio_path: str):
    """
    Identify speakers in an audio recording.

    Returns:
        A list of dictionaries containing start, end, and speaker.
    """
    audio_path = Path(audio_path).resolve()

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print(f"Input audio: {audio_path}")

    # Convert audio to mono, 16 kHz WAV for consistent processing.
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = str(Path(temp_dir) / "converted.wav")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", str(audio_path),
                    "-ac", "1",
                    "-ar", "16000",
                    "-c:a", "pcm_s16le",
                    wav_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"FFmpeg could not convert the audio:\n{exc.stderr}"
            ) from exc

        waveform, sample_rate = sf.read(
            wav_path,
            dtype="float32",
        )

    if waveform.size == 0:
        raise ValueError("The audio file is empty.")

    # Pyannote expects waveform shape: (channels, samples).
    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = np.nan_to_num(waveform)

    waveform_tensor = torch.from_numpy(
        waveform.copy()
    ).unsqueeze(0)

    print("\nLoading speaker diarization model...")

    pipeline = Pipeline.from_pretrained(MODEL_NAME)

    if pipeline is None:
        raise RuntimeError(
            "Could not load the diarization model. "
            "Check your Hugging Face access and authentication."
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        pipeline.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU. Processing may take longer.")

    print("\nIdentifying speakers...")

    output = pipeline(
        {
            "waveform": waveform_tensor,
            "sample_rate": sample_rate,
        }
    )

    diarization = output.exclusive_speaker_diarization

    segments = []

    for turn, speaker in diarization:
        if turn.end <= turn.start:
            continue

        segments.append(
            {
                "start": round(float(turn.start), 2),
                "end": round(float(turn.end), 2),
                "speaker": str(speaker),
            }
        )

    if not segments:
        print(
            "\nWarning: No speaker segments were detected."
        )
        return []

    # Sort segments chronologically.
    segments.sort(key=lambda item: item["start"])

    speakers = sorted(
        {segment["speaker"] for segment in segments}
    )

    print("\n========== DIARIZATION RESULTS ==========")
    print(f"Speaker labels detected: {speakers}")
    print(f"Number of distinct speakers: {len(speakers)}")
    print(f"Number of speech segments: {len(segments)}")

    for segment in segments:
        print(
            f"[{segment['start']:06.2f}s - "
            f"{segment['end']:06.2f}s] "
            f"{segment['speaker']}"
        )

    return segments


if __name__ == "__main__":
    audio_path = "uploads/college_project_meeting.wav"

    results = diarize_audio(audio_path)

    print("\n========== DIARIZATION SUMMARY ==========")

    speakers = sorted(
        {item["speaker"] for item in results}
    )

    print(f"Speakers detected: {speakers}")
    print(f"Total segments: {len(results)}")
