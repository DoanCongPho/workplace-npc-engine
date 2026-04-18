from agents.personas.base import PersonaConfig
from agents.personas.ceo import CEO_PERSONA
from agents.personas.chro import CHRO_PERSONA


PERSONAS: dict[str, PersonaConfig] = {
    CEO_PERSONA.persona_id: CEO_PERSONA,
    CHRO_PERSONA.persona_id: CHRO_PERSONA,
}


__all__ = ["PersonaConfig", "PERSONAS", "CEO_PERSONA", "CHRO_PERSONA"]
