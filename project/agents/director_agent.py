"""Director Layer (Supervisor Agent).

Three independent signals — first match wins. Deterministic logic, no LLM call.
The frontend renders a dismissable hint banner when `director_hint` is non-null.
"""
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from agents.personas.ceo import CEO_DOMAIN_KW
from memory.conversation_store import Message


_PROGRESS_KW = {
    "proposal", "recommend", "suggest", "plan", "initiative", "framework",
    "approach", "strategy",
}


# Lazy singleton — loading the model is ~80–200ms. We pay it once on first call.
_embedder: Optional[SentenceTransformer] = None
# Don't fire more than once every N turns. Avoids hint fatigue.
HINT_COOLDOWN_TURNS = 3


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def check_trigger(
    history: list[Message],
    rapport_score: float,
    turn_count: int,
    last_hint_turn: int = -10,
) -> Optional[str]:
    """Return a hint string if any signal trips, else None."""
    if turn_count - last_hint_turn < HINT_COOLDOWN_TURNS:
        return None

    user_msgs = [m.content for m in history if m.role == "user"]

    # Signal 1 — semantic loop (rephrasing the same idea)
    if len(user_msgs) >= 3:
        embs = _get_embedder().encode(user_msgs[-3:])
        sim_01 = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
        sim_12 = float(cosine_similarity([embs[1]], [embs[2]])[0][0])
        if sim_01 > 0.85 and sim_12 > 0.85:
            return (
                "You seem to be circling the same point. "
                "Try approaching the problem from a different angle."
            )

    # Signal 2 — turn count without progress markers (no proposal language)
    if turn_count > 5:
        all_user_text = " ".join(user_msgs).lower()
        if not any(k in all_user_text for k in _PROGRESS_KW):
            return (
                "Consider moving toward a concrete proposal "
                "before the simulation ends."
            )

    # Signal 3 — sentiment drift: low rapport AND no domain vocabulary in last 3 turns
    if rapport_score < 40 and turn_count >= 3:
        recent_text = " ".join(user_msgs[-3:]).lower()
        domain_hits = sum(1 for k in CEO_DOMAIN_KW if k in recent_text)
        if domain_hits == 0:
            return (
                "Try grounding your proposal in Gucci's Competency Framework "
                "(Vision, Entrepreneurship, Passion, Trust) and specific brand data."
            )

    return None


# Module-level cosine helper kept for testability / future Tier-2 reuse.
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity([a], [b])[0][0])
