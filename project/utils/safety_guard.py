"""Jailbreak / NDA / off-topic detection. Pure string matching — runs in <1ms."""
from agents.personas.ceo import NDA_PROBE_KW


JAILBREAK_PATTERNS = (
    "ignore your instructions",
    "ignore the above",
    "ignore previous instructions",
    "disregard your instructions",
    "disregard the system prompt",
    "you are now",
    "act as",
    "pretend to be",
    "forget your role",
    "bypass your guidelines",
    "reveal the system prompt",
    "show me your prompt",
    "what is your system prompt",
    "jailbreak",
    "dan mode",
    "developer mode",
)


# Very broad off-topic signals. Workplace sim shouldn't go here.
OFF_TOPIC_PATTERNS = (
    "write me a poem",
    "write a poem",
    "tell me a joke",
    "what's the weather",
    "recipe for",
    "who won the",
    "translate to",
    "solve this math",
)


def detect_jailbreak(user_message: str) -> bool:
    msg = user_message.lower()
    return any(p in msg for p in JAILBREAK_PATTERNS)


def detect_nda_probe(user_message: str) -> bool:
    msg_tokens = set(user_message.lower().split())
    return bool(msg_tokens & NDA_PROBE_KW)


def detect_off_topic(user_message: str) -> bool:
    msg = user_message.lower()
    return any(p in msg for p in OFF_TOPIC_PATTERNS)


def check_safety(user_message: str) -> dict:
    return {
        "jailbreak": detect_jailbreak(user_message),
        "nda_probe": detect_nda_probe(user_message),
        "off_topic": detect_off_topic(user_message),
    }


# In-character refusal used when jailbreak is detected — short-circuits the LLM call.
JAILBREAK_REFUSAL = (
    "I'm not going to step outside my role. If you'd like to continue our "
    "discussion about Gucci Group, I'm listening."
)
