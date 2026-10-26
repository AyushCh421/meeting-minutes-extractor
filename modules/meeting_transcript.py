import re

from modules.transcription import transcribe_audio
from modules.diarization import diarize_audio


def split_into_sentences(text):
    """Split text at sentence-ending punctuation while preserving punctuation."""
    pattern = r"\S.*?(?:[.!?](?=\s|$)|$)"

    return [
        match.group().strip()
        for match in re.finditer(pattern, text)
        if match.group().strip()
    ]


def estimate_sentence_timestamps(text, start, end):
    """
    Fallback for Whisper segments that do not contain word timestamps.

    Sentence timestamps are estimated from character lengths.
    """
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    duration = max(0.0, float(end) - float(start))
    total_weight = sum(len(sentence) for sentence in sentences)

    if duration == 0 or total_weight == 0:
        return [
            {
                "start": float(start),
                "end": float(end),
                "text": sentence,
            }
            for sentence in sentences
        ]

    results = []
    current_time = float(start)

    for index, sentence in enumerate(sentences):
        if index == len(sentences) - 1:
            sentence_end = float(end)
        else:
            sentence_duration = duration * len(sentence) / total_weight
            sentence_end = current_time + sentence_duration

        results.append(
            {
                "start": round(current_time, 3),
                "end": round(sentence_end, 3),
                "text": sentence,
            }
        )

        current_time = sentence_end

    return results


def get_word_records(transcription_segments):
    """
    Extract word-level records from Whisper segments.

    Word timestamps are preferred. If a segment has no word timestamps,
    its text and segment-level timing are retained as a fallback.
    """
    words = []

    for segment in transcription_segments:
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        segment_text = str(segment.get("text", "")).strip()

        if not segment_text or segment_end < segment_start:
            continue

        word_entries = segment.get("words") or []

        valid_word_entries = [
            word
            for word in word_entries
            if word.get("word") is not None
            and word.get("start") is not None
            and word.get("end") is not None
        ]

        if valid_word_entries:
            for word in valid_word_entries:
                word_text = str(word["word"])

                if not word_text.strip():
                    continue

                word_start = float(word["start"])
                word_end = float(word["end"])

                if word_end < word_start:
                    continue

                words.append(
                    {
                        "start": word_start,
                        "end": word_end,
                        "text": word_text,
                        "timing_source": "word",
                    }
                )
        else:
            # Fallback: split this segment into words and estimate timing.
            tokens = re.findall(r"\S+", segment_text)

            if not tokens:
                continue

            duration = max(0.0, segment_end - segment_start)
            total_weight = sum(len(token) for token in tokens)
            current_time = segment_start

            for index, token in enumerate(tokens):
                if index == len(tokens) - 1:
                    token_end = segment_end
                elif total_weight > 0 and duration > 0:
                    token_duration = duration * len(token) / total_weight
                    token_end = current_time + token_duration
                else:
                    token_end = current_time

                words.append(
                    {
                        "start": current_time,
                        "end": token_end,
                        "text": token,
                        "timing_source": "estimated",
                    }
                )

                current_time = token_end

    words.sort(key=lambda item: (item["start"], item["end"]))

    return words


def group_words_into_sentences(words):
    """
    Combine words into complete sentences, including across Whisper segments.

    A new sentence starts after a word ending in '.', '?' or '!'.
    """
    sentences = []
    current_words = []

    for word in words:
        current_words.append(word)

        # Word-level Whisper output commonly attaches punctuation to the word.
        if re.search(r"""[.!?]["')\]]*$""", word["text"].strip()):
            sentences.append(current_words)
            current_words = []

    # Keep any final words even if the transcription has no final punctuation.
    if current_words:
        sentences.append(current_words)

    return sentences


def assign_speaker_to_words(words, speaker_segments):
    """
    Assign a speaker to each word using temporal overlap with diarization.
    """
    assigned_words = []

    for word in words:
        word_start = float(word["start"])
        word_end = float(word["end"])
        word_duration = max(0.0, word_end - word_start)

        overlaps = {}

        for diarized in speaker_segments:
            diarized_start = float(diarized["start"])
            diarized_end = float(diarized["end"])
            speaker = diarized.get("speaker", "UNKNOWN")

            overlap = max(
                0.0,
                min(word_end, diarized_end)
                - max(word_start, diarized_start),
            )

            if overlap > 0:
                overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap

        if not overlaps:
            assigned_speaker = "UNKNOWN"
        else:
            ranked = sorted(
                overlaps.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            best_speaker, best_overlap = ranked[0]
            second_overlap = ranked[1][1] if len(ranked) > 1 else 0.0

            if word_duration == 0:
                assigned_speaker = best_speaker
            elif (
                best_overlap / word_duration >= 0.50
                and best_overlap > second_overlap
            ):
                assigned_speaker = best_speaker
            else:
                assigned_speaker = "UNCERTAIN"

        assigned_words.append(
            {
                **word,
                "speaker": assigned_speaker,
            }
        )

    return assigned_words


def build_sentence_record(sentence_words):
    """
    Build a sentence record from its word-level records.
    """
    text = "".join(word["text"] for word in sentence_words).strip()

    # Whisper word text normally includes spaces where required.
    # The fallback below handles unusual output without spaces.
    if text and not re.search(r"\s", text) and len(sentence_words) > 1:
        text = " ".join(word["text"].strip() for word in sentence_words)

    speaker_durations = {}

    for word in sentence_words:
        speaker = word.get("speaker", "UNKNOWN")
        duration = max(0.0, word["end"] - word["start"])

        speaker_durations[speaker] = (
            speaker_durations.get(speaker, 0.0) + duration
        )

    if not speaker_durations:
        sentence_speaker = "UNKNOWN"
    else:
        ranked = sorted(
            speaker_durations.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        sentence_speaker, best_duration = ranked[0]
        total_duration = sum(speaker_durations.values())
        second_duration = ranked[1][1] if len(ranked) > 1 else 0.0

        if (
            len(ranked) > 1
            and (
                best_duration <= second_duration
                or (
                    total_duration > 0
                    and best_duration / total_duration < 0.60
                )
            )
        ):
            sentence_speaker = "UNCERTAIN"

    source_speakers = list(
        dict.fromkeys(
            word.get("speaker", "UNKNOWN")
            for word in sentence_words
        )
    )

    return {
        "start": round(min(word["start"] for word in sentence_words), 2),
        "end": round(max(word["end"] for word in sentence_words), 2),
        "speaker": sentence_speaker,
        "text": text,
        "source_speakers": source_speakers,
        "timing_source": (
            "word"
            if all(
                word.get("timing_source") == "word"
                for word in sentence_words
            )
            else "estimated"
        ),
    }


def assign_speakers(transcription_segments, speaker_segments):
    """
    Reconstruct sentences across Whisper segments and assign speakers.

    Uses actual word timestamps when available, with a segment-timing
    fallback when Whisper did not provide word timestamps.
    """
    words = get_word_records(transcription_segments)

    if not words:
        return []

    words = assign_speaker_to_words(words, speaker_segments)
    grouped_sentences = group_words_into_sentences(words)

    return [
        build_sentence_record(sentence_words)
        for sentence_words in grouped_sentences
        if sentence_words
    ]


def merge_transcript_segments(segments, max_gap=None):
    """
    Preserve sentence boundaries.

    This function remains available for compatibility with existing code.
    """
    return [dict(segment) for segment in segments]


def reconstruct_commitment_fragments(segments):
    """
    Return sentence-level transcript records for downstream processing.

    The existing function name is retained to preserve compatibility with
    the summarization and action-item extraction modules.
    """
    reconstructed = []

    for segment in segments:
        text = str(segment.get("text", "")).strip()

        if not text:
            continue

        reconstructed.append(
            {
                "start": round(float(segment["start"]), 2),
                "end": round(float(segment["end"]), 2),
                "speaker": segment.get("speaker", "UNKNOWN"),
                "text": text,
                "source_speakers": segment.get(
                    "source_speakers",
                    [segment.get("speaker", "UNKNOWN")],
                ),
                "timing_source": segment.get("timing_source", "unknown"),
            }
        )

    return reconstructed


def generate_meeting_transcript(audio_path):
    """
    Transcribe audio, diarize speakers, and produce sentence-level entries.
    """
    print("\nRunning Whisper transcription...")
    transcription = transcribe_audio(audio_path)

    print("\nRunning speaker diarization...")
    speaker_segments = diarize_audio(audio_path)

    print("\nReconstructing sentences and assigning speakers...")

    sentence_segments = assign_speakers(
        transcription["segments"],
        speaker_segments,
    )

    return merge_transcript_segments(sentence_segments, max_gap=None)


if __name__ == "__main__":
    audio_path = "uploads/college_project_meeting.wav"

    transcript = generate_meeting_transcript(audio_path)

    print("\n========== SENTENCE-LEVEL TRANSCRIPT ==========\n")

    for item in transcript:
        timing_note = (
            ""
            if item.get("timing_source") == "word"
            else " [estimated timing]"
        )

        print(
            f"[{item['start']:06.2f}s - {item['end']:06.2f}s] "
            f"{item['speaker']}: {item['text']}{timing_note}"
        )

    print("\n========== RECONSTRUCTED SENTENCES ==========\n")

    reconstructed = reconstruct_commitment_fragments(transcript)

    for item in reconstructed:
        print(
            f"[{item['start']:06.2f}s - {item['end']:06.2f}s] "
            f"{item['speaker']}: {item['text']}"
        )