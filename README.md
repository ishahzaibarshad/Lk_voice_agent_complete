# LiveKit Room Voice Assistant

A compact Python voice assistant built on LiveKit Agents. It joins a LiveKit room and combines configurable OpenAI or Groq language-model support, Groq or OpenAI speech recognition, Cartesia speech synthesis, and Silero voice activity detection.

## What it does

- Runs as a LiveKit Agents worker and starts an assistant session in a room
- Uses OpenAI GPT-4.1 when an OpenAI key is configured, or Groq Llama when using the Groq path
- Uses Groq Whisper or OpenAI transcription for speech recognition, depending on configuration
- Produces audio with Cartesia TTS and applies Silero VAD
- Enables room transcription output

## Setup

Install dependencies with `pip install -r requirements.txt`. Configure the required LiveKit connection and provider API credentials in a local `.env` file. Never commit API keys or LiveKit secrets.

Run the worker in development mode:

```bash
python voice_agent.py dev
```

The repository also contains a backup agent script; `voice_agent.py` is the main entry point shown here.

## Stack

Python · LiveKit Agents · OpenAI · Groq · Cartesia · Silero VAD

