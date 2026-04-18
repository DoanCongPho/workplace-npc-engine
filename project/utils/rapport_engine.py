"""Rapport Engine — pure logic.

Engine knows nothing about specific personas. Each persona ships a
`RapportProfile` (tiers, weights, llm params) — the engine consumes it.

Two tiers:
- Tier 1 (rule-based, sync, <1ms) — fires every turn, drives the immediate tone.
- Tier 2 (LLM judge, async, ~300-500ms) — fires after the NPC has replied.
  Adjusts the score for the next turn. Semantic signals keywords miss.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from memory.conversation_store import RapportState

if TYPE_CHECKING:
    from agents.personas.base import PersonaConfig


logger = logging.getLogger(__name__)


# ---- Shared signal-extraction primitives (universal, not persona-specific) ----

STOPWORDS = {
    "the", "a", "an", "is", "are", "i", "you", "we", "to", "of", "and", "it",
    "in", "on", "for", "with", "this", "that", "be", "as", "at", "by",
}

POLITE_KW = {
    "please", "thank", "thanks", "i see", "that makes sense",
    "good point", "could you", "appreciate", "understood",
}


# ---- Persona-shaped configuration (consumed by the engine) -------------------

@dataclass(frozen=True)
class Tier:
    """A rapport tier — when state.score >= min_score, this tier is active."""
    min_score: float
    label: str            # "Frustrated" / "Engaged" / "Excited" / etc.
    tone_modifier: str    # directive injected into the system prompt
    llm_params: dict      # {"max_tokens": 50, "temperature": 0.3}


DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
    "length_ratio": 2.0,
    "politeness": 3.0,
    "vocab_mirror": 4.0,
    "domain_vocab": 5.0,
    "frustration": -8.0,
}


@dataclass(frozen=True)
class RapportProfile:
    """Per-persona rapport configuration."""
    tiers: tuple[Tier, ...]                                # MUST be sorted DESC by min_score
    starting_score: float = 50.0
    signal_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SIGNAL_WEIGHTS)
    )
    momentum_alpha: float = 0.7    # EMA weight on prior momentum
    momentum_factor: float = 0.2   # how much momentum nudges score per turn

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("RapportProfile.tiers must be non-empty")
        # Validate descending order — fail loudly at startup, not at first /chat
        sorted_desc = tuple(sorted(self.tiers, key=lambda t: -t.min_score))
        if sorted_desc != self.tiers:
            raise ValueError("RapportProfile.tiers must be sorted DESCENDING by min_score")


# ---- Engine functions (all persona-aware via RapportProfile) -----------------

def extract_signals(user_msg: str, npc_prev: str, persona_domain_kw: set) -> dict:
    """Five Tier-1 signals. Domain vocab uses the persona's own keyword set."""
    msg = user_msg.lower()
    npc_words = npc_prev.split()
    user_words = user_msg.split()

    length_ratio = min(len(user_words) / max(len(npc_words), 1), 2.0)

    politeness = sum(1 for p in POLITE_KW if p in msg) / len(POLITE_KW)

    npc_tokens = set(npc_prev.lower().split()) - STOPWORDS
    user_tokens = set(msg.split()) - STOPWORDS
    vocab_mirror = len(npc_tokens & user_tokens) / max(len(npc_tokens), 1)

    domain_hits = sum(1 for k in persona_domain_kw if k in msg)
    domain_vocab = min(domain_hits / 5.0, 1.0)

    frustration = 1.0 if len(user_words) < 5 else 0.0

    return {
        "length_ratio": length_ratio,
        "politeness": politeness,
        "vocab_mirror": vocab_mirror,
        "domain_vocab": domain_vocab,
        "frustration": frustration,
    }


def update_rapport(state: RapportState, signals: dict, profile: RapportProfile) -> RapportState:
    weights = profile.signal_weights
    delta = sum(weights.get(name, 0.0) * value for name, value in signals.items())
    state.momentum = (
        profile.momentum_alpha * state.momentum
        + (1.0 - profile.momentum_alpha) * delta
    )
    state.score = max(
        0.0,
        min(100.0, state.score + delta + profile.momentum_factor * state.momentum),
    )
    return state


def get_active_tier(state: RapportState, profile: RapportProfile) -> Tier:
    """Walk tiers (sorted descending) and return the first one matched."""
    for tier in profile.tiers:
        if state.score >= tier.min_score:
            return tier
    return profile.tiers[-1]


def get_tone_modifier(state: RapportState, profile: RapportProfile) -> str:
    return get_active_tier(state, profile).tone_modifier


def get_llm_params(state: RapportState, profile: RapportProfile) -> dict:
    return get_active_tier(state, profile).llm_params


def emotional_state_label(state: RapportState, profile: RapportProfile) -> str:
    return get_active_tier(state, profile).label


# ---- Tier 2: semantic LLM judge (async, runs AFTER the NPC response) ---------

class Tier2Signals(BaseModel):
    """LLM-scored nuance signals. 0.0 = absent, 1.0 = strongly present."""
    politeness: float = Field(
        ..., ge=0, le=1,
        description="Genuine warmth or respect toward the NPC. Semantic, not keyword-based.",
    )
    domain_depth: float = Field(
        ..., ge=0, le=1,
        description="Semantic engagement with the persona's professional domain. "
                    "Using concepts correctly counts — dropping keywords without understanding does not.",
    )
    strategic_substance: float = Field(
        ..., ge=0, le=1,
        description="Is the message a real contribution (proposal, observation, "
                    "counterargument) vs filler ('ok', 'guide me', 'idk')?",
    )
    frustration: float = Field(
        ..., ge=0, le=1,
        description="Visible frustration, disengagement, or giving up.",
    )
    reasoning: str = Field(..., description="One-sentence justification.")


# Tier 2 weights are modest on purpose — this is a semantic CORRECTION on top
# of Tier 1, not a replacement. It can shift score by roughly -4 to +8.5 per turn.
TIER2_WEIGHTS: dict[str, float] = {
    "politeness": 1.5,
    "domain_depth": 3.0,
    "strategic_substance": 4.0,
    "frustration": -4.0,
}


def tier2_delta(signals: Tier2Signals) -> float:
    return (
        TIER2_WEIGHTS["politeness"] * signals.politeness
        + TIER2_WEIGHTS["domain_depth"] * signals.domain_depth
        + TIER2_WEIGHTS["strategic_substance"] * signals.strategic_substance
        + TIER2_WEIGHTS["frustration"] * signals.frustration
    )


def apply_tier2(
    state: RapportState, signals: Tier2Signals, profile: RapportProfile
) -> RapportState:
    """Mutate state in place with Tier 2 semantic adjustment."""
    delta = tier2_delta(signals)
    state.momentum = (
        profile.momentum_alpha * state.momentum
        + (1.0 - profile.momentum_alpha) * delta
    )
    state.score = max(
        0.0,
        min(100.0, state.score + delta + profile.momentum_factor * state.momentum),
    )
    return state


_TIER2_PROMPT = """You are a silent rapport scorer observing a workplace \
simulation. You do NOT speak to the learner — you only score.

NPC role: {role_title}
Persona's domain vocabulary: {domain_kw}

NPC's prior response (for context):
\"\"\"{npc_msg}\"\"\"

Learner's message (score this):
\"\"\"{user_msg}\"\"\"

Score each axis 0.0 to 1.0:
- politeness: genuine respect/warmth. Absence of "thank" doesn't mean impolite; \
dismissive phrasing isn't polite even if formal.
- domain_depth: semantic engagement with the NPC's professional domain. Correct \
use of concepts counts; keyword-dropping does not.
- strategic_substance: real contribution (proposal, observation, counterargument) \
vs filler ("ok", "sure", "idk", "guide me").
- frustration: visible frustration, disengagement, or giving up.

Return JSON with four floats and one short reasoning sentence."""


def _build_tier2_prompt(user_msg: str, npc_msg: str, persona: "PersonaConfig") -> str:
    return _TIER2_PROMPT.format(
        role_title=persona.role_title,
        domain_kw=", ".join(sorted(persona.domain_kw)),
        npc_msg=npc_msg[:800],   # cap context to keep judge cheap
        user_msg=user_msg[:600],
    )


async def run_tier2_judge(
    user_msg: str,
    npc_msg: str,
    persona: "PersonaConfig",
    judge_llm,
) -> Optional[Tier2Signals]:
    """Call the LLM judge. Returns None on failure (caller should skip silently)."""
    prompt = _build_tier2_prompt(user_msg, npc_msg, persona)
    try:
        structured = judge_llm.with_structured_output(Tier2Signals)
        signals: Tier2Signals = await structured.ainvoke(prompt)
        return signals
    except Exception as exc:
        logger.warning("Tier 2 judge failed: %s", exc)
        return None
