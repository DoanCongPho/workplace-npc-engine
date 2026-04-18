import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models.schemas import ChatRequest, ChatResponse
from agents.npc_agent import NPCAgent
from memory.conversation_store import conversation_store

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="AI Co-Worker Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agents: dict[str, NPCAgent] = {}


def get_agent(persona_id: str) -> NPCAgent:
    if persona_id not in _agents:
        _agents[persona_id] = NPCAgent(persona_id=persona_id)
    return _agents[persona_id]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    agent = get_agent(req.persona_id)
    assistant_message, state, safety_flags = await agent.chat(
        persona_id=req.persona_id,
        user_message=req.user_message,
        session_id=req.session_id,
    )
    return ChatResponse(
        assistant_message=assistant_message,
        state=state,
        safety_flags=safety_flags,
    )


@app.post("/sessions/{session_id}/reset")
def reset_session(session_id: str) -> dict:
    conversation_store.reset(session_id)
    return {"reset": session_id}
