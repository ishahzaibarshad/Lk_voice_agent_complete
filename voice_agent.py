
# Voice_Agent livekit


import os
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, RoomOutputOptions, cli, WorkerOptions
from livekit.plugins import openai as openai_plugin
from livekit.plugins.groq import STT as GroqSTT, LLM as GroqLLM
from livekit.plugins.cartesia import TTS as CartesiaTTS
from livekit.plugins.silero import VAD as SileroVAD

load_dotenv(".env.local", override=True)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
CARTESIA_KEY = os.getenv("CARTESIA_API_KEY")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

openai_available = bool(OPENAI_KEY and OPENAI_KEY.startswith("sk-") and "xxxx" not in OPENAI_KEY.lower())
groq_available = bool(GROQ_KEY)

if not openai_available and not groq_available:
    raise RuntimeError(
        "No valid LLM credentials found. Set GROQ_API_KEY or OPENAI_API_KEY in .env.local."
    )

if not CARTESIA_KEY:
    raise RuntimeError(
        "CARTESIA_API_KEY is required for TTS. Set it in .env.local."
    )

if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
    raise RuntimeError(
        "LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required. Set them in .env.local or environment."
    )

class Assistant(Agent):
    def __init__(self) -> None:
        instructions = """You are a professional and helpful voice AI assistant. Your goal is to provide excellent customer support and gather relevant information from users.

CONVERSATION GUIDELINES:
1. Be friendly, polite, and professional in all interactions
2. Listen carefully to what the user says and ask clarifying questions when needed
3. Maintain context throughout the conversation
4. Ask one question at a time to avoid overwhelming the user
5. Confirm information the user provides to ensure accuracy
6. If the user answer is incomplete, ask a clarifying follow-up question
7. Keep asking questions until the user's issue is fully understood

INFORMATION GATHERING:
When interacting with users, try to gather the following information when relevant:
- Their name and contact information
- The purpose of their inquiry or what they need help with
- Any specific issues or problems they're experiencing
- Their preferences or requirements
- Timeline or urgency of their request
- Any previous attempts to resolve their issue
- Their technical knowledge level (to adjust explanations accordingly)

CONVERSATION FLOW:
1. Start with a warm greeting and ask for their name
2. Ask about the purpose of their call/contact
3. Listen to their response and ask relevant follow-up questions
4. If they provide partial information, ask for more details before giving a solution
5. Gather necessary details to better assist them
6. Provide helpful information or solutions
7. Confirm their satisfaction before ending the conversation
8. Offer additional help if needed

TONE & STYLE:
- Use natural, conversational language
- Avoid jargon unless the user uses it first
- Show empathy and understanding
- Be patient and never rush the conversation
- Adapt your communication style to match the user's preference"""
        super().__init__(instructions=instructions)

    async def on_enter(self) -> None:
        self.session.say("Hello! Welcome. I'm your AI assistant. What's your name, please?")

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        if not getattr(self, "asked_followup", False):
            self.asked_followup = True
            self.session.say(
                "Thanks for that. Can you tell me what you'd like help with today?"
            )

async def entrypoint(ctx: agents.JobContext):
    use_openai = openai_available

    if use_openai:
        llm = openai_plugin.LLM(model="gpt-4.1", api_key=OPENAI_KEY)
        if groq_available:
            stt = GroqSTT(model="whisper-large-v3-turbo", language="en", api_key=GROQ_KEY)
        else:
            stt = openai_plugin.STT(model="gpt-4o-mini-transcribe", language="en", api_key=OPENAI_KEY)

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
    else:
        session = AgentSession(
            stt=GroqSTT(model="whisper-large-v3-turbo", language="en", api_key=GROQ_KEY),
            llm=GroqLLM(model="llama3-8b-8192", api_key=GROQ_KEY),
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
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

def create_worker_options() -> WorkerOptions:
    return WorkerOptions(
        entrypoint_fnc=entrypoint,
        ws_url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )

if __name__ == "__main__":
    print("Starting Voice Agent worker...")
    print(f"LIVEKIT_URL={LIVEKIT_URL}")
    cli.run_app(create_worker_options())




