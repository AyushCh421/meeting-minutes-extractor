
import re
from transformers import pipeline


_summarizer = None


def load_summarizer():
    """Load the summarization model once."""
    global _summarizer

    if _summarizer is None:
        print("Loading summarization model...")
        _summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
        )

    return _summarizer


def clean_transcript(text):
    """Normalize whitespace and punctuation."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"\b(SPEAKER_\d+):\s*", r"\1: ", text)
    return text.strip()


def split_into_chunks(text, max_words=350):
    """Split transcript into chunks without cutting it at a fixed character count."""
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def summarize_chunk(text):
    """Summarize one chunk while avoiding overly long output."""
    words = text.split()

    if len(words) < 40:
        return text.strip()

    summarizer = load_summarizer()

    result = summarizer(
        text,
        max_length=min(110, max(45, len(words) - 5)),
        min_length=min(35, max(12, len(words) // 3)),
        do_sample=False,
        truncation=True,
    )

    return result[0]["summary_text"].strip()


def summarize_meeting(transcript: str) -> str:
    """Generate a concise, transcript-grounded meeting summary."""
    if not transcript or not transcript.strip():
        return "No transcript available to summarize."

    transcript = clean_transcript(transcript)
    chunks = split_into_chunks(transcript)

    summaries = [summarize_chunk(chunk) for chunk in chunks]
    summary = " ".join(summaries)

    # Clean common punctuation artifacts.
    summary = re.sub(r"\s+([.,!?])", r"\1", summary)
    summary = re.sub(r"\s+", " ", summary).strip()

    return summary


if __name__ == "__main__":
    sample = """
    SPEAKER_00: The product listing API is complete and connected
    to the database. The checkout button can be clicked twice, and
    the error message is difficult to read on smaller screens.
    SPEAKER_01: I will retest the cart count and double-click scenario
    when the new build is available. I will add regression tests for
    failed payments and empty address fields. I should have the results
    by Thursday.
    SPEAKER_02: I will upload the final mobile design by Tuesday evening.
    SPEAKER_00: I will contact Marketing about shipping and cancellation
    wording today and ask Operations for staging credentials.
    The release is targeted for Friday, conditional on successful
    checkout regression tests.
    """

    print("\n========== MEETING SUMMARY ==========\n")
    print(summarize_meeting(sample))
