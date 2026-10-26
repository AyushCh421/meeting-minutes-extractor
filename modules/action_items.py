import re


# --------------------------------------------------
# 1. DEADLINE DETECTION
# --------------------------------------------------

DEADLINE_PATTERNS = [
    (r"\bWednesday afternoon\b", "Wednesday afternoon"),
    (r"\bWednesday morning\b", "Wednesday morning"),
    (r"\bThursday afternoon\b", "Thursday afternoon"),
    (r"\bThursday morning\b", "Thursday morning"),
    (r"\bFriday afternoon\b", "Friday afternoon"),
    (r"\bFriday morning\b", "Friday morning"),
    (r"\bFriday evening\b", "Friday evening"),
    (r"\bSaturday afternoon\b", "Saturday afternoon"),
    (r"\bSaturday evening\b", "Saturday evening"),
    (r"\bMonday morning\b", "Monday morning"),
    (r"\bTuesday afternoon\b", "Tuesday afternoon"),
    (r"\bTuesday evening\b", "Tuesday evening"),
    (r"\bThursday\s+at\s+3\s*p\.?m\.?\b", "Thursday at 3 PM"),
    (r"\bFriday\s+at\s+4\s*p\.?m\.?\b", "Friday at 4 PM"),
    (r"\bby\s+tomorrow\b", "Tomorrow"),
    (r"\bby\s+today\b", "Today"),
    (r"\btoday\b", "Today"),
    (r"\btomorrow\b", "Tomorrow"),
    (r"\bnext Monday\b", "Next Monday"),
    (r"\bnext Friday\b", "Next Friday"),
    (r"\bMonday\b", "Monday"),
    (r"\bTuesday\b", "Tuesday"),
    (r"\bWednesday\b", "Wednesday"),
    (r"\bThursday\b", "Thursday"),
    (r"\bFriday\b", "Friday"),
    (r"\bSaturday\b", "Saturday"),
    (r"\bSunday\b", "Sunday"),
]


def normalize_text(text):
    """Normalize whitespace and surrounding punctuation."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text.strip(" \t\r\n.,;-")


def extract_deadline(text):
    """Return the first recognized deadline."""
    for pattern, label in DEADLINE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return label

    return "Not specified"


# --------------------------------------------------
# 2. SENTENCE SPLITTING
# --------------------------------------------------

def split_into_sentences(text):
    """Split text into individual sentences."""
    text = normalize_text(text)

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        normalize_text(sentence)
        for sentence in sentences
        if normalize_text(sentence)
    ]


# --------------------------------------------------
# 3. ACTION DETECTION PATTERNS
# --------------------------------------------------

# Explicit commitments:
# "I will complete the API."
# "We'll send the report."
# "I should have the report ready."

COMMITMENT_PATTERN = re.compile(
    r"\b(?:"
    r"I\s+will|I'll|"
    r"We\s+will|We'll|"
    r"I\s+shall|We\s+shall|"
    r"I\s+should\s+have|We\s+should\s+have"
    r")\b",
    re.IGNORECASE,
)


# Obligations and recommendations:
# "We need to arrange the room."
# "We should confirm the projector."
# "The team must finish testing."
# "We have to deploy the application."

OBLIGATION_PATTERN = re.compile(
    r"\b(?:"
    r"we|I|you|they|he|she|"
    r"the team|the developer|the manager|"
    r"someone|everybody|everyone"
    r")\s+"
    r"(?:"
    r"need(?:s)?\s+to|"
    r"should|"
    r"must|"
    r"have\s+to|"
    r"has\s+to|"
    r"ought\s+to|"
    r"are\s+required\s+to|"
    r"is\s+required\s+to"
    r")\s+",
    re.IGNORECASE,
)


# Requests and instructions:
# "Please send the report."
# "Could you update the documentation?"
# "Can you test the API?"
# "Make sure to check the deployment."

REQUEST_PATTERN = re.compile(
    r"^(?:"
    r"please\s+"
    r"|could\s+you\s+"
    r"|can\s+you\s+"
    r"|would\s+you\s+"
    r"|kindly\s+"
    r"|make\s+sure\s+(?:to|that)\s+"
    r"|remember\s+to\s+"
    r"|don't\s+forget\s+to\s+"
    r"|ensure\s+that\s+"
    r")",
    re.IGNORECASE,
)


def is_actionable_commitment(text):
    """
    Determine whether a sentence expresses a commitment,
    obligation, recommendation, request, or instruction.
    """
    text = normalize_text(text)

    if not text:
        return False

    # Reject short fragments that are unlikely to be complete tasks.
    if len(text.split()) < 4:
        return False

    if COMMITMENT_PATTERN.search(text):
        return True

    if OBLIGATION_PATTERN.search(text):
        return True

    if REQUEST_PATTERN.search(text):
        return True

    return False


# --------------------------------------------------
# 4. CLASSIFY ACTION TYPE
# --------------------------------------------------

def classify_action_type(text):
    """Classify an action as a commitment, obligation, or request."""
    if COMMITMENT_PATTERN.search(text):
        return "Commitment"

    if OBLIGATION_PATTERN.search(text):
        return "Obligation / Recommendation"

    if REQUEST_PATTERN.search(text):
        return "Request / Instruction"

    return "Action"


# --------------------------------------------------
# 5. SPLIT COORDINATED COMMITMENTS
# --------------------------------------------------

def split_commitments(text):
    """
    Split separately stated commitments.

    Example:
        "I will update the docs and I will push the code."

    Produces:
        "I will update the docs"
        "I will push the code"

    Ordinary coordinated actions remain together.
    """
    patterns = [
        r"\s*,?\s+and\s+(?="
        r"(?:I\s+will|I'll|We\s+will|We'll)\b)",

        r"\s*,?\s+and\s+(?="
        r"(?:we|I|you|they|the team|someone)\s+"
        r"(?:need\s+to|needs\s+to|should|must|have\s+to)\b)",

        r"\s*,?\s+and\s+(?=please\s+)",
    ]

    parts = [normalize_text(text)]

    for pattern in patterns:
        new_parts = []

        for part in parts:
            new_parts.extend(
                re.split(pattern, part, flags=re.IGNORECASE)
            )

        parts = [
            normalize_text(part)
            for part in new_parts
            if normalize_text(part)
        ]

    return parts


# --------------------------------------------------
# 6. TASK CLEANING AND DUPLICATE DETECTION
# --------------------------------------------------

def clean_task(text):
    """Clean surrounding whitespace and punctuation."""
    return normalize_text(text)


def make_task_key(task, speaker):
    """Create a key for avoiding duplicate tasks."""
    normalized = task.lower()

    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        normalized,
    ).strip()

    return normalized, speaker


# --------------------------------------------------
# 7. MAIN ACTION-ITEM EXTRACTION
# --------------------------------------------------

def extract_action_items(transcript):
    """
    Extract action items from transcript segments.

    Each segment should contain:
        start, end, speaker, text

    Returns a list of dictionaries containing:
        task
        assignee
        deadline
        type
        status

    IMPORTANT:
    The speaker label is not necessarily the real task owner.
    The initial status is a default, not a verified task status.
    """
    action_items = []
    seen_tasks = set()

    for segment in transcript:
        text = clean_task(segment.get("text", ""))
        speaker = segment.get("speaker") or "UNKNOWN"

        if not text:
            continue

        # Step A: Split the text into sentences.
        sentences = split_into_sentences(text)

        for sentence in sentences:

            # Step B: Split separately stated commitments.
            parts = split_commitments(sentence)

            for part in parts:
                part = clean_task(part)

                # Step C: Check whether the text is actionable.
                if not is_actionable_commitment(part):
                    continue

                # Step D: Avoid duplicate tasks from the same speaker.
                task_key = make_task_key(part, speaker)

                if task_key in seen_tasks:
                    continue

                seen_tasks.add(task_key)

                # Step E: Classify and save the task.
                action_items.append({
                    "task": part,
                    "assignee": speaker,
                    "deadline": extract_deadline(part),
                    "type": classify_action_type(part),
                    "status": "Pending",
                })

    return action_items


# --------------------------------------------------
# 8. QUICK TEST
# --------------------------------------------------

if __name__ == "__main__":

    sample_transcript = [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "SPEAKER_00",
            "text": (
                "I will complete the attendance API "
                "by Wednesday afternoon."
            ),
        },
        {
            "start": 5.0,
            "end": 9.0,
            "speaker": "SPEAKER_01",
            "text": "We should confirm the room and projector.",
        },
        {
            "start": 9.0,
            "end": 13.0,
            "speaker": "SPEAKER_01",
            "text": "We need to arrange the meeting room.",
        },
        {
            "start": 13.0,
            "end": 17.0,
            "speaker": "SPEAKER_00",
            "text": "Please send the report by Friday.",
        },
        {
            "start": 17.0,
            "end": 20.0,
            "speaker": "SPEAKER_00",
            "text": "The meeting is finished. Thank you, everyone.",
        },
        {
            "start": 20.0,
            "end": 24.0,
            "speaker": "SPEAKER_01",
            "text": "I have finished the login page.",
        },
    ]

    results = extract_action_items(sample_transcript)

    print("\n========== ACTION ITEM TEST ==========\n")

    if results:
        for index, item in enumerate(results, start=1):
            print(f"{index}. Task: {item['task']}")
            print(f"   Assignee: {item['assignee']}")
            print(f"   Deadline: {item['deadline']}")
            print(f"   Type: {item['type']}")
            print(f"   Status: {item['status']}")
            print()
    else:
        print("No action items detected.")