# LiveKit Room Voice Assistant

A small Python voice assistant worker for LiveKit rooms. It uses Cartesia for speech synthesis, Silero for voice activity detection, and either OpenAI or Groq for language and speech recognition.

## Implemented provider paths

- With `OPENAI_API_KEY` set, the worker uses OpenAI GPT-4.1 for the LLM. If `GROQ_API_KEY` is also set, it uses Groq Whisper for STT; otherwise it uses OpenAI transcription.
- With only `GROQ_API_KEY` set, it uses the Groq LLM and Groq Whisper STT.
- `CARTESIA_API_KEY` is required for TTS.
- The agent greets the room participant and publishes transcription output.

The worker connects to a LiveKit server; it is not an independent telephony service and does not place outbound phone calls.

## Requirements

- Python 3.10 or newer
- A reachable LiveKit server and its URL/API key/API secret
- A Cartesia key and at least one of an OpenAI or Groq key

The included `livekit-server/docker-compose.yml` starts a local development server. Its development mode and exposed media ports are for a trusted local setup only; do not expose it to the public internet or reuse development credentials in a deployed service.

## Install and configure

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set the provider and LiveKit values in `.env`. For a local LiveKit development server, start it from the repository root with:

```bash
docker compose -f livekit-server/docker-compose.yml up
```

The local `--dev` server uses development-only settings. For LiveKit Cloud, use the URL and credentials from your own project.

## Run

With the server available and `.env` configured:

```bash
python voice_agent.py dev
```

A participant must join a LiveKit room to exercise the audio conversation. The repository does not include a deployment or public demo.

## Tests

The tests validate provider-key selection and missing-key errors without loading LiveKit plugins or contacting model APIs:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

## Security and cost notes

- `.env` is ignored by Git; never commit provider credentials.
- The worker needs valid LiveKit credentials at runtime. Keep local development-server credentials on a trusted machine.
- OpenAI, Groq, and Cartesia usage may incur provider charges. Configure your own limits and review current provider pricing before running live sessions.
- Provider, room, network, and audio behavior require external services and are not exercised by the unit tests.

## Repository contents

```text
voice_agent.py                 Main worker
voice_agent_backup.py          Alternate/backup worker
config.py                      Provider key validation
livekit-server/docker-compose.yml  Optional local development server
sip.yaml                       SIP configuration example

