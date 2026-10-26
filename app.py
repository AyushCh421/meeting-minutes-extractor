
import re
import sys
from pathlib import Path

import streamlit as st

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.meeting_transcript import (
    generate_meeting_transcript,
    reconstruct_commitment_fragments,
)
from modules.summarization import summarize_meeting
from modules.action_items import extract_action_items


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Meeting Minutes Extractor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎙️ Automated Meeting Minutes Extractor")

st.markdown(
    """
    Transform meeting recordings into structured meeting minutes
    using speech recognition, speaker diarization, and NLP.

    Upload a recording, review the transcript, correct recognition
    errors, and generate your final meeting report.
    """
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("📌 About This Project")

    st.write(
        """
        Automatically transcribe meeting recordings, identify speakers,
        summarize discussions, and extract action items.
        """
    )

    st.subheader("Features")

    st.markdown(
        """
        - 🎧 Whisper speech transcription
        - 👥 Pyannote speaker diarization
        - ✏️ Editable transcript
        - 📝 Meeting summarization
        - ✅ Action-item extraction
        - 📅 Deadline identification
        - 🏷️ Task classification
        - 📄 Timestamped transcript
        - 📥 Downloadable meeting minutes
        """
    )

    st.divider()

    st.caption("Automated Meeting Minutes & Action Item Extractor")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def format_timestamped_transcript(transcript):
    """Format transcript segments with timestamps and speaker labels."""

    lines = []

    for segment in transcript:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        speaker = segment.get("speaker", "UNKNOWN")
        text = str(segment.get("text", "")).strip()

        if text:
            lines.append(
                f"[{start:.2f}s - {end:.2f}s] "
                f"{speaker}: {text}"
            )

    return "\n".join(lines)


def parse_edited_transcript(edited_text):
    """
    Parse the editable timestamped transcript.

    Expected format:
    [0.00s - 4.25s] SPEAKER_00: Example sentence.
    """

    segments = []
    errors = []

    pattern = re.compile(
        r"^\s*\[\s*(\d+(?:\.\d+)?)s\s*-\s*"
        r"(\d+(?:\.\d+)?)s\s*\]\s*"
        r"([A-Za-z0-9_-]+)\s*:\s*(.*?)\s*$"
    )

    for line_number, line in enumerate(
        edited_text.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        match = pattern.match(line)

        if not match:
            errors.append(
                f"Line {line_number}: Invalid format."
            )
            continue

        start = float(match.group(1))
        end = float(match.group(2))
        speaker = match.group(3)
        text = match.group(4).strip()

        if start < 0:
            errors.append(
                f"Line {line_number}: Start time cannot be negative."
            )
            continue

        if end <= start:
            errors.append(
                f"Line {line_number}: End time must be greater than start time."
            )
            continue

        if not text:
            errors.append(
                f"Line {line_number}: Transcript text cannot be empty."
            )
            continue

        segments.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            }
        )

    return segments, errors


def create_meeting_report(
    summary,
    action_items,
    transcript_text,
):
    """Build a downloadable plain-text meeting report."""

    report_lines = [
        "=" * 60,
        "MEETING MINUTES",
        "=" * 60,
        "",
        "MEETING SUMMARY",
        "-" * 60,
        str(summary),
        "",
        "ACTION ITEMS",
        "-" * 60,
    ]

    if action_items:
        for index, item in enumerate(action_items, start=1):
            report_lines.extend(
                [
                    "",
                    f"{index}. Task: {item.get('task', 'Not specified')}",
                    f"   Assignee: {item.get('assignee', 'Unknown')}",
                    f"   Deadline: {item.get('deadline', 'Not specified')}",
                    f"   Type: {item.get('type', 'Unknown')}",
                    f"   Status: {item.get('status', 'Pending')}",
                ]
            )
    else:
        report_lines.append("No actionable tasks were detected.")

    report_lines.extend(
        [
            "",
            "FULL TRANSCRIPT",
            "-" * 60,
            transcript_text,
            "",
            "=" * 60,
            "END OF MEETING MINUTES",
            "=" * 60,
        ]
    )

    return "\n".join(report_lines)


def generate_minutes_from_transcript(transcript):
    """Generate summary and action items from transcript segments."""

    reconstructed = reconstruct_commitment_fragments(transcript)

    if not reconstructed:
        raise ValueError(
            "No usable transcript sentences were found."
        )

    transcript_for_summary = " ".join(
        str(segment.get("text", "")).strip()
        for segment in reconstructed
        if str(segment.get("text", "")).strip()
    )

    if not transcript_for_summary:
        raise ValueError("The transcript is empty.")

    summary = summarize_meeting(transcript_for_summary)
    action_items = extract_action_items(reconstructed)

    return summary, action_items


def save_results(summary, action_items, transcript_text):
    """Save results in Streamlit session state and to the outputs folder."""

    report_text = create_meeting_report(
        summary=summary,
        action_items=action_items,
        transcript_text=transcript_text,
    )

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "meeting_minutes.txt"

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    st.session_state["meeting_summary"] = summary
    st.session_state["meeting_action_items"] = action_items
    st.session_state["meeting_transcript"] = transcript_text
    st.session_state["meeting_report"] = report_text
    st.session_state["meeting_report_path"] = str(report_path)


# --------------------------------------------------
# AUDIO UPLOAD
# --------------------------------------------------

st.subheader("📤 Upload Meeting Recording")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["mp3", "wav", "m4a", "ogg", "flac"],
    help="Supported formats: MP3, WAV, M4A, OGG, and FLAC.",
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    st.success(f"Audio uploaded: {uploaded_file.name}")

    if st.button(
        "🎙️ Transcribe Meeting",
        type="primary",
        use_container_width=True,
    ):
        upload_dir = BASE_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = Path(uploaded_file.name).name
        audio_path = upload_dir / safe_filename

        try:
            audio_path.write_bytes(uploaded_file.getvalue())

            with st.spinner(
                "Transcribing audio and identifying speakers..."
            ):
                transcript = generate_meeting_transcript(
                    str(audio_path)
                )

            if not transcript:
                st.error(
                    "No transcript was generated. "
                    "Please check your audio recording."
                )
            else:
                transcript_text = format_timestamped_transcript(
                    transcript
                )

                # Keep the original transcription for comparison.
                st.session_state["original_transcript"] = transcript_text
                st.session_state["editable_transcript"] = transcript_text
                st.session_state["transcript_ready"] = True
                st.session_state["source_filename"] = safe_filename

                # Clear old results when a new recording is transcribed.
                for key in [
                    "meeting_summary",
                    "meeting_action_items",
                    "meeting_transcript",
                    "meeting_report",
                    "meeting_report_path",
                ]:
                    st.session_state.pop(key, None)

                st.success(
                    "Transcription complete! "
                    "Review and correct it below."
                )

        except Exception as error:
            st.error("An error occurred during transcription.")
            st.exception(error)


# --------------------------------------------------
# EDITABLE TRANSCRIPT
# --------------------------------------------------

if st.session_state.get("transcript_ready", False):
    st.divider()
    st.header("✏️ Review and Edit Transcript")

    st.markdown(
        """
        Correct misrecognized words before generating your minutes.

        **Important:** Keep each line in this format:

        `[0.00s - 5.20s] SPEAKER_00: Your transcript text`

        You can edit the words, but preserve the timestamps and speaker
        labels unless you have verified that they are incorrect.
        """
    )

    with st.expander("View original transcription"):
        st.text_area(
            "Original transcript",
            value=st.session_state["original_transcript"],
            height=250,
            disabled=True,
            key="original_transcript_view",
        )

    st.text_area(
        "Editable timestamped transcript",
        key="editable_transcript",
        height=350,
        help=(
            "Correct words directly in this box. "
            "Preserve the timestamp and speaker label on every line."
        ),
    )

    if st.button(
        "✅ Generate Minutes from Edited Transcript",
        type="primary",
        use_container_width=True,
    ):
        edited_text = st.session_state.get(
            "editable_transcript",
            "",
        )

        transcript_segments, errors = parse_edited_transcript(
            edited_text
        )

        if errors:
            st.error(
                "Please fix the following transcript formatting errors:"
            )

            for error in errors:
                st.write(f"- {error}")

        elif not transcript_segments:
            st.error("The edited transcript is empty.")

        else:
            try:
                with st.spinner(
                    "Generating summary and extracting action items..."
                ):
                    summary, action_items = (
                        generate_minutes_from_transcript(
                            transcript_segments
                        )
                    )

                corrected_transcript = format_timestamped_transcript(
                    transcript_segments
                )

                save_results(
                    summary=summary,
                    action_items=action_items,
                    transcript_text=corrected_transcript,
                )

                st.success(
                    "Meeting minutes generated from the edited transcript!"
                )

            except Exception as error:
                st.error(
                    "Could not generate meeting minutes from the transcript."
                )
                st.exception(error)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "meeting_report" in st.session_state:
    st.divider()
    st.header("📊 Generated Meeting Minutes")

    summary = st.session_state["meeting_summary"]
    action_items = st.session_state["meeting_action_items"]
    transcript_text = st.session_state["meeting_transcript"]
    report_text = st.session_state["meeting_report"]

    unique_speakers = set()

    for segment in transcript_text.splitlines():
        match = re.search(
            r"\]\s+([A-Za-z0-9_-]+):",
            segment,
        )

        if match:
            unique_speakers.add(match.group(1))

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Action Items",
        len(action_items),
    )

    metric2.metric(
        "Speakers Detected",
        len(unique_speakers),
    )

    metric3.metric(
        "Transcript Characters",
        len(transcript_text),
    )

    tab_summary, tab_actions, tab_transcript, tab_download = st.tabs(
        [
            "📝 Meeting Summary",
            "✅ Action Items",
            "📄 Full Transcript",
            "📥 Download Report",
        ]
    )

    with tab_summary:
        st.subheader("Meeting Summary")
        st.write(summary)

    with tab_actions:
        st.subheader("Extracted Action Items")

        if action_items:
            action_table = [
                {
                    "Task": item.get("task", ""),
                    "Assignee": item.get("assignee", "Unknown"),
                    "Deadline": item.get("deadline", "Not specified"),
                    "Type": item.get("type", "Unknown"),
                    "Status": item.get("status", "Pending"),
                }
                for item in action_items
            ]

            st.dataframe(
                action_table,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No actionable tasks were detected.")

    with tab_transcript:
        st.subheader("Final Timestamped Transcript")

        st.text_area(
            "Transcript",
            value=transcript_text,
            height=450,
            disabled=True,
            key="final_transcript_view",
        )

    with tab_download:
        st.subheader("Download Meeting Minutes")

        st.download_button(
            label="📥 Download Meeting Minutes (.txt)",
            data=report_text,
            file_name="meeting_minutes.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.caption(
            "The report contains the summary, action items, deadlines, "
            "task classifications, statuses, and final transcript."
        )

        st.info("A copy is also saved in your project's outputs folder.")

        st.code(
            st.session_state.get("meeting_report_path", ""),
            language=None,
        )

else:
    st.info(
        "Upload a meeting recording, click 'Transcribe Meeting', "
        "review the transcript, and then generate the meeting minutes."
    )
