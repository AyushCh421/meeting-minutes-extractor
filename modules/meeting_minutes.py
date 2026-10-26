
from pathlib import Path

from modules.meeting_transcript import (
    generate_meeting_transcript,
    reconstruct_commitment_fragments,
)
from modules.summarization import summarize_meeting
from modules.action_items import extract_action_items


def format_action_item(index, item):
    """Format an action item for the report and terminal."""
    return [
        "",
        f"{index}. Task: {item.get('task', 'Not specified')}",
        f"   Assignee: {item.get('assignee', 'Unknown')}",
        f"   Deadline: {item.get('deadline', 'Not specified')}",
        f"   Type: {item.get('type', 'Action')}",
        f"   Status: {item.get('status', 'Pending')}",
    ]


def main():
    base_dir = Path(__file__).resolve().parent.parent

    audio_path = base_dir / "uploads" / "college_project_meeting.wav"
    output_dir = base_dir / "outputs"
    report_path = output_dir / "meeting_minutes2.txt"

    print("\n" + "=" * 60)
    print("MEETING MINUTES EXTRACTOR")
    print("=" * 60)

    if not audio_path.exists():
        print(f"\nERROR: Audio file not found: {audio_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Transcribe and diarize.
    print("\n[1/3] Transcribing audio and identifying speakers...")

    transcript = generate_meeting_transcript(str(audio_path))

    if not transcript:
        print("\nNo transcript was generated.")
        return

    # Step 2: Reconstruct complete sentences.
    print("\nReconstructing sentences across transcript segments...")

    reconstructed = reconstruct_commitment_fragments(transcript)

    if not reconstructed:
        print("\nNo usable transcript sentences were found.")
        return

    summary_input = " ".join(
        item["text"] for item in reconstructed
    )

    # Step 3: Generate the meeting summary.
    print("\n[2/3] Generating meeting summary...")

    summary = summarize_meeting(summary_input)

    # Step 4: Extract action items from reconstructed sentences.
    print("\n[3/3] Extracting action items...")

    action_items = extract_action_items(reconstructed)

    # Step 5: Build the report.
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
            report_lines.extend(format_action_item(index, item))
    else:
        report_lines.append("No action items detected.")

    report_lines.extend([
        "",
        "FULL TRANSCRIPT",
        "-" * 60,
    ])

    # Keep the original segmented transcript for the report.
    for item in transcript:
        text = item.get("text", "").strip()

        if text:
            report_lines.append(
                f"[{item['start']:.2f}s - {item['end']:.2f}s] "
                f"{item.get('speaker', 'UNKNOWN')}: {text}"
            )

    report = "\n".join(report_lines)

    # Display summary.
    print("\n" + "=" * 60)
    print("MEETING SUMMARY")
    print("=" * 60)
    print(summary)

    # Display action items.
    print("\n" + "=" * 60)
    print("ACTION ITEMS")
    print("=" * 60)

    if action_items:
        for index, item in enumerate(action_items, start=1):
            print("\n" + "\n".join(format_action_item(index, item)))
    else:
        print("No action items detected.")

    # Save the report.
    report_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print("MEETING MINUTES GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nReport saved to:\n{report_path}")


if __name__ == "__main__":
    main()
