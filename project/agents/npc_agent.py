"""Core NPC Agent — LangGraph StateGraph with 6 nodes.

Pipeline (per turn):
  safety_check → retrieve_context → director → build_prompt → llm_call → update_state

Key design decisions:
  - Safety uses gpt-4o-mini semantic classification; the result is a targeted
    instruction injected at the TOP of the system prompt so the main LLM
    generates an in-character, tier-aware response (not a canned string).
  - Director runs BEFORE build_prompt so its coaching hint is also woven into
    the system prompt and the NPC delivers it naturally.
  - last_hint_turn is persisted in ConversationStore so the cooldown works
    correctly across turns.

Persona-agnostic: every persona-specific behaviour is loaded from PERSONAS[id].
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agents.director_agent import check_trigger
from agents.personas import PERSONAS, PersonaConfig
from memory.conversation_store import Message, conversation_store
from memory.vector_store import vector_store
from models.schemas import StateUpdate
from utils.rapport_engine import (
    apply_tier2,
    emotional_state_label,
    extract_signals,
    get_llm_params,
    get_tone_modifier,
    run_tier2_judge,
    tier2_delta,
    update_rapport,
)
from utils.safety_guard import run_safety_check

logger = logging.getLogger(__name__)


class NPCState(TypedDict):
    session_id: str
    persona_id: str
    user_message: str
    conversation_history: list[Message]
    retrieved_context: str
    assistant_message: str
    rapport_score: float
    rapport_momentum: float
    emotional_state: str
    turn_count: int
    safety_flags: dict
    safety_injection: str          # prepended to system prompt when non-empty
    director_hint: Optional[str]
    last_hint_turn: int
    messages: list


def _persona_for(state: NPCState) -> PersonaConfig:
    return PERSONAS[state["persona_id"]]


# --- Nodes ----------------------------------------------------------------

RAG_SCORE_THRESHOLD = 0.55
RAG_MIN_WORDS = 4


def _retrieve_context_node(state: NPCState) -> NPCState:
    persona = _persona_for(state)
    msg = state["user_message"]

    if len(msg.split()) < RAG_MIN_WORDS:
        state["retrieved_context"] = ""
        state["safety_flags"]["rag_skipped"] = "too_short"
        return state

    try:
        idx = vector_store.load_or_build(persona.knowledge_file, persona.persona_id)
        docs, top_score = idx.search_with_score(msg, k=3)
        state["safety_flags"]["rag_top_score"] = round(top_score, 3)

        if top_score < RAG_SCORE_THRESHOLD:
            state["retrieved_context"] = ""
            state["safety_flags"]["rag_skipped"] = "below_threshold"
            return state

        state["retrieved_context"] = "\n\n".join(
            f"- {d.title}: {d.content}" for d in docs
        )
    except Exception as e:
        state["retrieved_context"] = ""
        state["safety_flags"]["rag_error"] = str(e)
    return state


def _director_node(state: NPCState) -> NPCState:
    """Run BEFORE build_prompt so the coaching hint is woven into the system prompt."""
    persona = _persona_for(state)
    history = state["conversation_history"] + [
        Message(role="user", content=state["user_message"]),
    ]
    hint = check_trigger(
        history=history,
        rapport_score=state["rapport_score"],
        turn_count=state["turn_count"],
        persona=persona,
        last_hint_turn=state.get("last_hint_turn", -10),
    )
    state["director_hint"] = hint
    if hint:
        conversation_store.set_last_hint_turn(state["session_id"], state["turn_count"])
    return state


def _build_prompt_node(state: NPCState) -> NPCState:
    persona = _persona_for(state)
    rapport_state = conversation_store.get_rapport(
        state["session_id"], starting_score=persona.rapport_profile.starting_score
    )
    tone_modifier = get_tone_modifier(rapport_state, persona.rapport_profile)

    system_text = persona.system_prompt.format(
        rapport_tone_modifier=tone_modifier,
        retrieved_context=state["retrieved_context"] or "(no specific context retrieved)",
    )

    # Prepend safety injection — LLM generates an in-character response to
    # jailbreaks, NDA probes, and off-topic questions instead of a canned string.
    if state.get("safety_injection"):
        system_text = state["safety_injection"] + system_text

    # Inject director coaching cue so NPC weaves it naturally into the reply.
    if state.get("director_hint"):
        system_text += (
            "\n\n[INTERNAL COACHING CUE — do not quote this note directly, "
            "but let it subtly shape your response to help the learner progress]\n"
            f"{state['director_hint']}"
        )

    messages: list = [SystemMessage(content=system_text)]
    for m in state["conversation_history"][-10:]:
        if m.role == "user":
            messages.append(HumanMessage(content=m.content))
        else:
            messages.append(AIMessage(content=m.content))
    messages.append(HumanMessage(content=state["user_message"]))

    state["messages"] = messages
    return state


def _update_state_node(state: NPCState) -> NPCState:
    persona = _persona_for(state)
    history = state["conversation_history"]
    npc_prev = next(
        (m.content for m in reversed(history) if m.role == "assistant"),
        "",
    )
    signals = extract_signals(
        user_msg=state["user_message"],
        npc_prev=npc_prev,
        persona_domain_kw=set(persona.domain_kw),
    )
    rapport_state = conversation_store.get_rapport(
        state["session_id"], starting_score=persona.rapport_profile.starting_score
    )
    rapport_state = update_rapport(rapport_state, signals, persona.rapport_profile)
    conversation_store.set_rapport(state["session_id"], rapport_state)

    state["rapport_score"] = rapport_state.score
    state["rapport_momentum"] = rapport_state.momentum
    state["emotional_state"] = emotional_state_label(rapport_state, persona.rapport_profile)

    state["safety_flags"].update({
        "rapport_score": rapport_state.score,
        "momentum": rapport_state.momentum,
        "is_stuck": signals["frustration"] > 0 and state["turn_count"] > 3,
        "frustration_risk": rapport_state.score < 30 and rapport_state.momentum < -2.0,
        "engagement_peak": rapport_state.score > 80,
    })
    return state


# --- Agent ----------------------------------------------------------------

class NPCAgent:
    def __init__(self, persona_id: str = "ceo"):
        if persona_id not in PERSONAS:
            raise ValueError(f"Unknown persona: {persona_id}")
        self.persona: PersonaConfig = PERSONAS[persona_id]
        self.persona_id = persona_id
        self._llm: Optional[ChatOpenAI] = None
        self._judge_llm: Optional[ChatOpenAI] = None
        self.graph = self._build_graph()

    @property
    def llm(self) -> Optional[ChatOpenAI]:
        if self._llm is None and os.environ.get("OPENAI_API_KEY"):
            self._llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        return self._llm

    @property
    def judge_llm(self) -> Optional[ChatOpenAI]:
        if self._judge_llm is None and os.environ.get("OPENAI_API_KEY"):
            self._judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        return self._judge_llm

    async def _safety_check_node(self, state: NPCState) -> NPCState:
        """Async safety check — uses gpt-4o-mini for semantic classification.

        Result is a targeted instruction stored in state["safety_injection"].
        build_prompt prepends it to the system prompt so the main LLM generates
        a fully in-character, tier-aware response.
        """
        persona = _persona_for(state)
        flags = await run_safety_check(
            user_message=state["user_message"],
            persona=persona,
            llm=self.judge_llm,   # None when no API key → keyword-only fallback
        )
        state["safety_flags"] = {**state.get("safety_flags", {}), **flags}
        state["safety_injection"] = flags.get("safety_injection", "")

        # No-LLM fallback: if API key absent AND obvious jailbreak, use canned string.
        if self.llm is None and flags.get("jailbreak") and flags.get("safety_phase") == "keyword":
            state["assistant_message"] = persona.jailbreak_refusal

        return state

    async def _tier2_update(
        self, session_id: str, user_msg: str, npc_msg: str,
    ) -> None:
        if self.judge_llm is None:
            return
        signals = await run_tier2_judge(user_msg, npc_msg, self.persona, self.judge_llm)
        if signals is None:
            return
        s = conversation_store.get_rapport(
            session_id, starting_score=self.persona.rapport_profile.starting_score
        )
        before = s.score
        apply_tier2(s, signals, self.persona.rapport_profile)
        conversation_store.set_rapport(session_id, s)
        logger.info(
            "tier2 session=%s delta=%.2f score=%.1f→%.1f reason=%s",
            session_id, tier2_delta(signals), before, s.score, signals.reasoning,
        )

    def _llm_node(self, state: NPCState) -> NPCState:
        # No-LLM jailbreak was already handled in _safety_check_node.
        if state.get("assistant_message"):
            return state

        messages = state.get("messages", [])
        if self.llm is None:
            state["assistant_message"] = (
                "[Configuration: OPENAI_API_KEY not set. Safety, rapport, "
                "Director, and RAG layers are still active — see right panel.]"
            )
            return state

        persona = _persona_for(state)
        rapport_state = conversation_store.get_rapport(
            state["session_id"], starting_score=persona.rapport_profile.starting_score
        )
        params = get_llm_params(rapport_state, persona.rapport_profile)
        try:
            res = self.llm.invoke(
                messages,
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
            )
            state["assistant_message"] = res.content
        except Exception as e:
            state["assistant_message"] = f"[LLM error: {e}]"
            state["safety_flags"]["llm_error"] = str(e)
        return state

    def _build_graph(self):
        g = StateGraph(NPCState)
        g.add_node("safety_check", self._safety_check_node)   # async method
        g.add_node("retrieve_context", _retrieve_context_node)
        g.add_node("director", _director_node)
        g.add_node("build_prompt", _build_prompt_node)
        g.add_node("llm_call", self._llm_node)
        g.add_node("update_state", _update_state_node)
        g.add_edge(START, "safety_check")
        g.add_edge("safety_check", "retrieve_context")
        g.add_edge("retrieve_context", "director")
        g.add_edge("director", "build_prompt")
        g.add_edge("build_prompt", "llm_call")
        g.add_edge("llm_call", "update_state")
        g.add_edge("update_state", END)
        return g.compile()

    async def chat(
        self, persona_id: str, user_message: str, session_id: str
    ) -> tuple[str, StateUpdate, dict]:
        history = list(conversation_store.get_history(session_id))
        rapport_st = conversation_store.get_rapport(
            session_id, starting_score=self.persona.rapport_profile.starting_score
        )

        initial: NPCState = {
            "session_id": session_id,
            "persona_id": persona_id,
            "user_message": user_message,
            "conversation_history": history,
            "retrieved_context": "",
            "assistant_message": "",
            "rapport_score": rapport_st.score,
            "rapport_momentum": rapport_st.momentum,
            "emotional_state": emotional_state_label(rapport_st, self.persona.rapport_profile),
            "turn_count": len(history) // 2 + 1,
            "safety_flags": {},
            "safety_injection": "",
            "director_hint": None,
            "last_hint_turn": conversation_store.get_last_hint_turn(session_id),
            "messages": [],
        }
        result = await self.graph.ainvoke(initial)

        conversation_store.append(session_id, "user", user_message)
        conversation_store.append(session_id, "assistant", result["assistant_message"])

        # Tier 2 — skip on safety blocks and errors
        safe = result["safety_flags"].get("safety_category", "safe") == "safe"
        if safe and not result["safety_flags"].get("llm_error"):
            asyncio.create_task(
                self._tier2_update(
                    session_id=session_id,
                    user_msg=user_message,
                    npc_msg=result["assistant_message"],
                )
            )

        state_update = StateUpdate(
            rapport_score=result["rapport_score"],
            rapport_momentum=result["rapport_momentum"],
            emotional_state=result["emotional_state"],
            turn_count=result["turn_count"],
            director_hint=result["director_hint"],
        )
        return result["assistant_message"], state_update, result["safety_flags"]
