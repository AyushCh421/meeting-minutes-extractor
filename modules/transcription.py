import whisper


def transcribe_audio(audio_path, model_name="small"):
    """
    Transcribe an audio file using Whisper.

    Returns:
        transcript: Complete transcription text.
        segments: Whisper segments with word-level timestamps.
        language: Detected language.
    """
    print(f"Loading Whisper model: {model_name}...")

    model = whisper.load_model(model_name)

    print("Transcribing audio...")

    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        fp16=False,
        task="transcribe",
        temperature=0,
        condition_on_previous_text=True,
        initial_prompt=(
            "This is a software product development meeting. "
            "Topics include a product listing API, checkout, "
            "payment failures, duplicate payment requests, "
            "idempotency checks, cart updates, regression tests, "
            "mobile UI design, API documentation, staging credentials, "
            "shipping text, cancellation policy, and a Friday launch. "
            "Preserve technical terminology, names, dates, and deadlines."
        ),
    )

    return {
        "transcript": result["text"].strip(),
        "segments": result["segments"],
        "language": result.get("language"),
    }


if __name__ == "__main__":
    audio_path = "uploads/college_project_meeting.wav"

    result = transcribe_audio(audio_path)

    print("\n========== TRANSCRIPT ==========\n")
    print(result["transcript"])

    print("\nLanguage:", result["language"])

    print("\n========== WORD-LEVEL TIMESTAMPS ==========\n")

    for segment in result["segments"]:
        for word in segment.get("words", []):
            print(
                f"[{word['start']:06.2f}s - "
                f"{word['end']:06.2f}s] "
                f"{word['word']}"
            )