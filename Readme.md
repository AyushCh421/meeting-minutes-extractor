# 🎙️ AI Meeting Minutes & Action Item Extractor

An AI-powered application that transforms meeting recordings into structured transcripts, concise summaries, and actionable tasks. It combines speech recognition, speaker diarization, and natural language processing to reduce the effort required to document meetings.

## ✨ Features

- **Audio Transcription:** Convert recorded meetings into text using OpenAI Whisper.
- **Speaker Diarization:** Identify and distinguish speakers using Pyannote Audio.
- **Meeting Summarization:** Generate concise summaries using a transformer-based summarization model.
- **Action Item Extraction:** Identify tasks, responsible speakers, deadlines, and task status.
- **Interactive Interface:** Upload meeting recordings and access results through Streamlit.
- **Structured Output:** Generate meeting minutes that are easier to review and share.

## 🛠️ Tech Stack

- **Language:** Python
- **Interface:** Streamlit
- **Speech Recognition:** OpenAI Whisper
- **Speaker Diarization:** Pyannote Audio
- **NLP & Summarization:** Hugging Face Transformers
- **Environment Management:** Python virtual environment

## 📂 Project Structure

```text
meeting-minutes-extractor/
├── app.py
├── modules/
│   ├── __init__.py
│   ├── audio_processing.py
│   ├── transcription.py
│   ├── diarization.py
│   ├── summarization.py
│   ├── action_items.py
│   ├── meeting_transcript.py
│   └── meeting_minutes.py
├── uploads/
├── outputs/
├── tests/
├── test_transcription.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/AyushCh421/meeting-minutes-extractor.git
cd meeting-minutes-extractor
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

FFmpeg must also be installed and available on your system PATH for audio processing.

### 4. Configure environment variables

Create a `.env` file in the project root if the application requires environment variables.

For speaker diarization, you may need a Hugging Face access token and access to the relevant gated model. Store the token in your local environment configuration and never commit it to GitHub.

Example:

```env
HF_TOKEN=your_huggingface_token
```

Use the environment-variable name expected by your code if it differs from `HF_TOKEN`.

### 5. Run the application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Follow the instructions displayed in the application to upload a supported meeting recording and process it.

## 🔄 Processing Pipeline

1. Upload a meeting recording.
2. Process the audio and prepare it for transcription.
3. Generate a transcript using Whisper.
4. Identify speakers using Pyannote Audio.
5. Generate a summary using a transformer-based NLP model.
6. Extract action items and relevant deadlines.
7. Review the generated meeting minutes.

## 🔐 Security and Privacy

- Keep API tokens and other credentials out of source control.
- Do