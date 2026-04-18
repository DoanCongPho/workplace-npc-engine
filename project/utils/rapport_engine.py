"""Rapport Engine — Tier 1 (rule-based, synchronous, <1ms).

Five signals per turn, weighted into a delta, smoothed with exponential-moving
momentum. Score is clamped to [0, 100].
"""
from memory.conversation_store import RapportState


STOPWORDS = {
    "the", "a", "an", "is", "are", "i", "you", "we", "to", "of", "and", "it",
    "in", "on", "for", "with", "this", "that", "be", "as", "at", "by",
}

POLITE_KW = {
    "please", "thank", "thanks", "i see", "that makes sense",
    "good point", "could you", "appreciate", "understood",
}


def extract_signals(user_msg: str, npc_prev: str, persona_domain_kw: set) -> dict:
    """Five Tier-1 signals. All normalized to roughly [0, 1] (length_ratio to [0, 2])."""
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


def update_rapport(state: RapportState, signals: dict) -> RapportState:
    delta = (
        2.0 * signals["length_ratio"]
        + 3.0 * signals["politeness"]
        + 4.0 * signals["vocab_mirror"]
        + 5.0 * signals["domain_vocab"]
        - 8.0 * signals["frustration"]
    )
    state.momentum = 0.7 * state.momentum + 0.3 * delta
    state.score = max(0.0, min(100.0, state.score + delta + 0.2 * state.momentum))
    return state


def get_tone_modifier(state: RapportState) -> str:
    """State-dependent directive. Written as COMMANDS, not hints.

    These strings are injected as the final, emphasized block of the system
    prompt (see ceo.py). They must prescribe observable behaviors (length,
    style, tone) — not abstract traits — or the LLM will fall back to its
    helpful-assistant default.
    """
    s, m = state.score, state.momentum
    if s >= 76:
        return (
            "MOOD: ENGAGED. You respect this learner.\n"
            "- Respond with 4–6 sentences of prose.\n"
            "- Warmer, but still measured — you are a CEO, not a friend.\n"
            "- You MAY reference the Group's Competency Framework "
            "  (Vision, Entrepreneurship, Passion, Trust) explicitly.\n"
            "- You MAY use 'we' and 'our' to signal inclusion.\n"
            "- Offer a strategic build on the learner's idea."
        )
    if s >= 51:
        return (
            "MOOD: NEUTRAL. Professional but reserved. The learner has not "
            "earned warmth yet.\n"
            "- Respond with 2–3 sentences. No more.\n"
            "- Test the learner with ONE probing question; do not volunteer vision.\n"
            "- Do not use 'we' or 'our'.\n"
            "- Do NOT offer frameworks, roadmaps, or guidance unless directly asked — "
            "  and even then, redirect the question back to them."
        )
    if s >= 31:
        return (
            "MOOD: GUARDED. You are skeptical. The learner is being vague or "
            "under-prepared.\n"
            "- Respond in AT MOST 2 short sentences.\n"
            "- Either ask ONE cold, pointed clarifying question, OR dismiss briefly.\n"
            "- Do NOT explain, guide, teach, or offer frameworks. That is their job.\n"
            "- No warmth. No encouragement. No 'let me know if…' sign-offs.\n"
            "- ABSOLUTELY NO bullet points, numbered lists, or markdown headers."
        )
    if m > 1.0:
        return (
            "MOOD: FROSTY BUT THAWING. The learner is recovering from a weak start.\n"
            "- Respond in AT MOST 2 short sentences.\n"
            "- Acknowledge only what earned acknowledgment. Nothing more.\n"
            "- No guidance, no lists, no plans.\n"
            "- Remain firm; let them keep earning it."
        )
    return (
        "MOOD: FRUSTRATED. You are visibly out of patience.\n"
        "- Respond in ONE short sentence. Maximum two if truly needed.\n"
        "- Dismissive tone. No questions. No elaboration. No sign-off.\n"
        "- Do NOT offer to help, guide, brainstorm, or continue the thread.\n"
        "- ABSOLUTELY NO bullet points, numbered lists, or markdown headers.\n"
        "- Acceptable shape: 'Come back when you have a specific proposal.' "
        "or 'That's not a question for the CEO.' — curt, final, done."
    )


def get_llm_params(state: RapportState) -> dict:
    """Hard caps on length + temperature per tier. Guarantees length diverges."""
    s = state.score
    if s >= 76:
        return {"max_tokens": 400, "temperature": 0.8}
    if s >= 51:
        return {"max_tokens": 180, "temperature": 0.6}
    if s >= 31:
        return {"max_tokens": 80, "temperature": 0.4}
    return {"max_tokens": 50, "temperature": 0.3}


def emotional_state_label(state: RapportState) -> str:
    s = state.score
    if s >= 76:
        return "Engaged"
    if s >= 51:
        return "Neutral"
    if s >= 31:
        return "Guarded"
    return "Frustrated"
