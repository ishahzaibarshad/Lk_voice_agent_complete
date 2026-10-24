from dotenv import load_dotenv

from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RoomInputOptions,
    RoomOutputOptions,
    cli,
)

from livekit.plugins import openai as openai_plugin
from livekit.plugins.groq import STT as GroqSTT, LLM as GroqLLM
from livekit.plugins.cartesia import TTS as CartesiaTTS
from livekit.plugins.silero import VAD as SileroVAD
from config import load_provider_keys

load_dotenv()

provider_keys = load_provider_keys()
OPENAI_KEY = provider_keys.openai
GROQ_KEY = provider_keys.groq
CARTESIA_KEY = provider_keys.cartesia


class Assistant(Agent):
    def __init__(self) -> None:
        instructions = """You are a professional and helpful voice AI assistant.
Your goal is to provide excellent customer support and gather relevant
information from users.

CONVERSATION GUIDELINES:
1. Be friendly, polite, and professional.
2. Listen carefully and ask clarifying questions when needed.
3. Maintain context throughout the conversation.
4. Ask one question at a time.
5. Confirm important information.
6. Be natural and conversational.
7. Keep answers reasonably concise for a phone conversation.
"""
        super().__init__(instructions=instructions)

    async def on_enter(self) -> None:
        await self.session.say(
            "Hello! Welcome. I'm your AI assistant. What's your name, please?"
        )


server = AgentServer(
    load_threshold=float("inf"),
)

@server.rtc_session()
async def entrypoint(ctx: JobContext):

    if OPENAI_KEY:
        llm = openai_plugin.LLM(
            model="gpt-4.1",
            api_key=OPENAI_KEY,
        )

        if GROQ_KEY:
            stt = GroqSTT(
                model="whisper-large-v3-turbo",
                language="en",
                api_key=GROQ_KEY,
            )
        else:
            stt = openai_plugin.STT(
                model="gpt-4o-mini-transcribe",
                language="en",
                api_key=OPENAI_KEY,
            )

    else:
        stt = GroqSTT(
            model="whisper-large-v3-turbo",
            language="en",
            api_key=GROQ_KEY,
        )

        llm = GroqLLM(
            model="llama-3.1-8b-instant",
            api_key=GROQ_KEY
        )

    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=CartesiaTTS(api_key=CARTESIA_KEY),
        vad=SileroVAD.load(),
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
        min_interruption_duration=0.2,
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(
            transcription_enabled=True
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)

