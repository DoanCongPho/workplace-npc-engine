"""Core NPC Agent — LangGraph StateGraph with 6 nodes.

Pipeline (per turn):
  safety_check → retrieve_context → build_prompt → llm_call → update_state → director
"""
from __future__ import annotations

import os
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agents.director_agent import check_trigger
from agents.personas.ceo import PERSONAS, PersonaConfig
from memory.conversation_store import Message, conversation_store
from memory.vector_store import vector_store
from models.schemas import StateUpdate
from utils.rapport_engine import (
    emotional_state_label,
    extract_signals,
    get_llm_params,
    get_tone_modifier,
    update_rapport,
)
from utils.safety_guard import (
    JAILBREAK_REFUSAL,
    check_safety,
)


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
    director_hint: Optional[str]
    last_hint_turn: int
    messages: list


# --- Nodes ----------------------------------------------------------------

def _safety_check_node(state: NPCState) -> NPCState:
    flags = check_safety(state["user_message"])
    state["safety_flags"] = {**state.get("safety_flags", {}), **flags}
    return state


def _retrieve_context_node(state: NPCState) -> NPCState:
    persona = PERSONAS[state["persona_id"]]
    msg_tokens = set(state["user_message"].lower().split())
    if not (msg_tokens & persona.knowledge_trigger_kw):
        state["retrieved_context"] = ""
        return state
    try:
        idx = vector_store.load_or_build(persona.knowledge_file, persona.persona_id)
        docs = idx.search(state["user_message"], k=3)
        state["retrieved_context"] = "\n\n".join(
            f"- {d.title}: {d.content}" for d in docs
        )
    except Exception as e:
        # RAG failure must not break the conversation.
        state["retrieved_context"] = ""
        state["safety_flags"]["rag_error"] = str(e)
    return state


def _build_prompt_node(state: NPCState) -> NPCState:
    persona = PERSONAS[state["persona_id"]]
    rapport_state = conversation_store.get_rapport(state["session_id"])
    tone_modifier = get_tone_modifier(rapport_state) or "Professional baseline."

    system_text = persona.system_prompt.format(
        rapport_tone_modifier=tone_modifier,
        retrieved_context=state["retrieved_context"] or "(no specific context retrieved)",
    )

    messages: list = [SystemMessage(content=system_text)]
    for m in state["conversation_history"][-10:]:  # cap history sent to LLM
        if m.role == "user":
            messages.append(HumanMessage(content=m.content))
        else:
            messages.append(AIMessage(content=m.content))
    messages.append(HumanMessage(content=state["user_message"]))

    state["messages"] = messages
    return state


def _update_state_node(state: NPCState) -> NPCState:
    persona = PERSONAS[state["persona_id"]]
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
    rapport_state = conversation_store.get_rapport(state["session_id"])
    rapport_state = update_rapport(rapport_state, signals)
    conversation_store.set_rapport(state["session_id"], rapport_state)

    state["rapport_score"] = rapport_state.score
    state["rapport_momentum"] = rapport_state.momentum
    state["emotional_state"] = emotional_state_label(rapport_state)

    state["safety_flags"].update({
        "rapport_score": rapport_state.score,
        "momentum": rapport_state.momentum,
        "is_stuck": signals["frustration"] > 0 and state["turn_count"] > 3,
        "frustration_risk": rapport_state.score < 30 and rapport_state.momentum < -2.0,
        "engagement_peak": rapport_state.score > 80,
    })
    return state


def _director_node(state: NPCState) -> NPCState:
    # We don't persist last_hint_turn yet — cooldown is best-effort within session.
    history = state["conversation_history"] + [
        Message(role="user", content=state["user_message"]),
        Message(role="assistant", content=state["assistant_message"]),
    ]
    hint = check_trigger(
        history=history,
        rapport_score=state["rapport_score"],
        turn_count=state["turn_count"],
        last_hint_turn=state.get("last_hint_turn", -10),
    )
    state["director_hint"] = hint
    return state


# --- Agent ----------------------------------------------------------------

class NPCAgent:
    def __init__(self, persona_id: str = "ceo"):
        if persona_id not in PERSONAS:
            raise ValueError(f"Unknown persona: {persona_id}")
        self.persona: PersonaConfig = PERSONAS[persona_id]
        self.persona_id = persona_id
        self._llm: Optional[ChatOpenAI] = None
        self.graph = self._build_graph()

    @property
    def llm(self) -> Optional[ChatOpenAI]:
        if self._llm is None and os.environ.get("OPENAI_API_KEY"):
            self._llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        return self._llm

    def _llm_node(self, state: NPCState) -> NPCState:
        # Short-circuit: jailbreak attempt → in-character refusal, skip LLM.
        if state["safety_flags"].get("jailbreak"):
            state["assistant_message"] = JAILBREAK_REFUSAL
            return state

        messages = state.get("messages", [])
        if self.llm is None:
            state["assistant_message"] = (
                "[Configuration: OPENAI_API_KEY not set. Other engine layers "
                "(safety, rapport, Director, RAG) are still running — see right panel.]"
            )
            return state

        rapport_state = conversation_store.get_rapport(state["session_id"])
        params = get_llm_params(rapport_state)
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
        g.add_node("safety_check", _safety_check_node)
        g.add_node("retrieve_context", _retrieve_context_node)
        g.add_node("build_prompt", _build_prompt_node)
        g.add_node("llm_call", self._llm_node)
        g.add_node("update_state", _update_state_node)
        g.add_node("director", _director_node)
        g.add_edge(START, "safety_check")
        g.add_edge("safety_check", "retrieve_context")
        g.add_edge("retrieve_context", "build_prompt")
        g.add_edge("build_prompt", "llm_call")
        g.add_edge("llm_call", "update_state")
        g.add_edge("update_state", "director")
        g.add_edge("director", END)
        return g.compile()

    async def chat(
        self, persona_id: str, user_message: str, session_id: str
    ) -> tuple[str, StateUpdate, dict]:
        history = list(conversation_store.get_history(session_id))
        rapport_st = conversation_store.get_rapport(session_id)

        initial: NPCState = {
            "session_id": session_id,
            "persona_id": persona_id,
            "user_message": user_message,
            "conversation_history": history,
            "retrieved_context": "",
            "assistant_message": "",
            "rapport_score": rapport_st.score,
            "rapport_momentum": rapport_st.momentum,
            "emotional_state": emotional_state_label(rapport_st),
            "turn_count": len(history) // 2 + 1,
            "safety_flags": {},
            "director_hint": None,
            "last_hint_turn": -10,
            "messages": [],
        }
        result = await self.graph.ainvoke(initial)

        conversation_store.append(session_id, "user", user_message)
        conversation_store.append(session_id, "assistant", result["assistant_message"])

        state_update = StateUpdate(
            rapport_score=result["rapport_score"],
            rapport_momentum=result["rapport_momentum"],
            emotional_state=result["emotional_state"],
            turn_count=result["turn_count"],
            director_hint=result["director_hint"],
        )
        return result["assistant_message"], state_update, result["safety_flags"]
