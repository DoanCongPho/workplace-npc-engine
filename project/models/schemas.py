from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    persona_id: str = Field(default="ceo")
    session_id: str
    user_message: str


class StateUpdate(BaseModel):
    rapport_score: float = 50.0
    rapport_momentum: float = 0.0
    emotional_state: str = "Neutral"
    turn_count: int = 0
    director_hint: Optional[str] = None


class ChatResponse(BaseModel):
    assistant_message: str
    state: StateUpdate
    safety_flags: dict = {}
