"""In-memory per-session history + rapport state. Replace with Redis later."""
from dataclasses import dataclass, field
from typing import Literal


Role = Literal["user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class RapportState:
    score: float = 50.0
    momentum: float = 0.0


class ConversationStore:
    def __init__(self) -> None:
        self._history: dict[str, list[Message]] = {}
        self._rapport: dict[str, RapportState] = {}
        self._last_hint_turn: dict[str, int] = {}

    def get_history(self, session_id: str) -> list[Message]:
        return self._history.setdefault(session_id, [])

    def append(self, session_id: str, role: Role, content: str) -> None:
        self.get_history(session_id).append(Message(role=role, content=content))

    def get_rapport(self, session_id: str, starting_score: float = 50.0) -> RapportState:
        return self._rapport.setdefault(session_id, RapportState(score=starting_score))

    def set_rapport(self, session_id: str, state: RapportState) -> None:
        self._rapport[session_id] = state

    def get_last_hint_turn(self, session_id: str) -> int:
        return self._last_hint_turn.get(session_id, -10)

    def set_last_hint_turn(self, session_id: str, turn: int) -> None:
        self._last_hint_turn[session_id] = turn

    def reset(self, session_id: str) -> None:
        self._history.pop(session_id, None)
        self._rapport.pop(session_id, None)
        self._last_hint_turn.pop(session_id, None)


conversation_store = ConversationStore()
