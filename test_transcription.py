from modules.transcription import transcribe_audio

audio_path = "uploads/sample_meeting_3_minutes.mp3"

result = transcribe_audio(audio_path)

print("\n========== TRANSCRIPTION ==========")
print(result["transcript"])

print("\n========== LANGUAGE ==========")
print(result["language"])

print("\n========== TIMESTAMPED SEGMENTS ==========")

for segment in result["segments"]:
    print(
        f"[{segment['start']:.2f}s - "
        f"{segment['end']:.2f}s] "
        f"{segment['text']}"
    )