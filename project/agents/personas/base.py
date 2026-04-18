"""Persona base type — shared by all NPCs.

Adding a new co-worker = create one of these instances. No engine code edits.
"""
from dataclasses import dataclass

from utils.rapport_engine import RapportProfile


@dataclass(frozen=True)
class PersonaConfig:
    persona_id: str
    name: str
    role_title: str
    system_prompt: str                  # MUST contain {rapport_tone_modifier} and {retrieved_context}
    domain_kw: frozenset[str]           # used by rapport (domain_vocab signal) and Director Signal 3
    nda_probe_kw: frozenset[str]        # confidentiality keywords specific to this role
    knowledge_file: str                 # path relative to project root
    rapport_profile: RapportProfile
    jailbreak_refusal: str              # fallback only when OPENAI_API_KEY absent + keyword jailbreak detected
