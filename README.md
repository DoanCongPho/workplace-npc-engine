# AI Co-Worker Engine

A conversational simulation prototype where learners practise high-stakes workplace conversations with AI-powered NPC executives. Built for the Edtronaut AI Engineer Intern take-home assignment.

---

## What it does

You chat with a senior executive (Group CEO or Group CHRO of Gucci Group). The NPC's tone, verbosity, and willingness to engage adapt in real time based on how well you communicate — strategic depth, domain vocabulary, politeness, and engagement all shift a **rapport score** that drives the conversation forward or pushes back.

A **Director layer** watches for learner stalls (semantic loops, lack of progress, topic drift) and injects a coaching nudge delivered in the NPC's own voice. A **safety layer** handles jailbreaks and NDA probes in-character via LLM semantic classification — no canned refusal strings.

---

## Architecture

```
Browser (React + Tailwind)
        │  POST /chat
        ▼
FastAPI  ──────────────────────────────────────────────────────┐
        │                                                      │
        ▼                                                      │
LangGraph StateGraph (per turn)                        ConversationStore
  safety_check   ← gpt-4o-mini semantic classifier     (history, rapport,
       │                                                last_hint_turn)
  retrieve_context ← FAISS + nomic-embed-text (RAG)
       │
  director       ← all-MiniLM-L6-v2 (loop / stall / drift)
       │
  build_prompt   ← weaves safety_injection + director_hint into system prompt
       │
  llm_call       ← gpt-4o (temperature + max_tokens set by rapport tier)
       │
  update_state   ← Tier-1 rapport update (sync, <1 ms)
       │
  [async]        ← Tier-2 rapport judge (gpt-4o-mini, fire-and-forget)
```

---

## Project structure

```
.
├── project/                  # Python backend
│   ├── main.py               # FastAPI app + routes
│   ├── agents/
│   │   ├── npc_agent.py      # LangGraph pipeline (6 nodes)
│   │   ├── director_agent.py # Director: loop/stall/drift signals
│   │   └── personas/
│   │       ├── base.py       # PersonaConfig dataclass
│   │       ├── ceo.py        # Lorenzo Bertelli — CEO persona
│   │       └── chro.py       # Group CHRO persona
│   ├── memory/
│   │   ├── conversation_store.py  # In-memory history + rapport state
│   │   └── vector_store.py        # FAISS index builder/loader
│   ├── models/
│   │   └── schemas.py        # Pydantic request/response models
│   ├── utils/
│   │   ├── rapport_engine.py # Two-tier rapport logic
│   │   └── safety_guard.py   # Two-phase safety classifier
│   ├── data/
│   │   ├── ceo_knowledge.json
│   │   └── chro_knowledge.json
│   ├── requirements.txt
│   └── .env.example
├── frontend/                 # React + TypeScript + Tailwind
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── SimHeader.tsx         # Persona switcher tabs
│       │   ├── ChatWindow.tsx        # Message bubbles + input
│       │   ├── StatePanel.tsx        # Rapport bar + tier badge
│       │   └── DirectorHintBanner.tsx
│       ├── hooks/
│       │   └── useConversation.ts
│       └── lib/
│           └── personas.ts
└── docs/
    ├── docs.tex              # Full technical report (LaTeX)
    └── docs.pdf              # Compiled 8-page report
```

---

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 18+
- An OpenAI API key (GPT-4o access required)
- [Ollama](https://ollama.com) with `nomic-embed-text` pulled (for RAG embeddings)

```bash
ollama pull nomic-embed-text
```

### 1. Backend

```bash
cd project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

uvicorn main:app --reload --port 8000
```

Backend is now running at `http://localhost:8000`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## API

### `POST /chat`

```json
{
  "persona_id": "ceo",
  "session_id": "abc-123",
  "user_message": "I think we should unify brand identities across the portfolio."
}
```

Response:

```json
{
  "assistant_message": "...",
  "state": {
    "rapport_score": 54.2,
    "rapport_momentum": 1.8,
    "emotional_state": "Neutral",
    "turn_count": 3,
    "director_hint": null
  },
  "safety_flags": {
    "safety_category": "safe",
    "rag_top_score": 0.71
  }
}
```

### `POST /sessions/{session_id}/reset`

Clears conversation history and rapport state for the session.

### `GET /health`

Returns `{"status": "ok"}`.

---

## Personas

| ID | Name | Starting rapport | Tiers |
|----|------|-----------------|-------|
| `ceo` | Lorenzo Bertelli, Group CEO | 50 | Frustrated → Guarded → Neutral → Engaged |
| `chro` | Group CHRO | 60 | Concerned → Supportive → Energized |

Adding a new persona requires only a new `PersonaConfig` instance — no engine code changes.

---

## Key design decisions

**Rapport is two-tier.**
Tier-1 fires synchronously every turn using five rule-based signals (message length ratio, politeness keywords, vocabulary mirroring, domain keyword density, frustration markers). Tier-2 fires asynchronously after the NPC replies — a `gpt-4o-mini` judge scores the exchange for nuance the rules miss and adjusts the score for the next turn.

**Safety is in-character.**
A two-phase classifier runs before the main LLM. Phase 1 is a fast keyword pre-filter (<1 ms). Phase 2 calls `gpt-4o-mini` to semantically classify the message (`safe` / `jailbreak` / `nda_probe` / `off_topic`). The result is a targeted instruction prepended to the system prompt — the main `gpt-4o` generates a persona-appropriate, rapport-tier-aware refusal, not a hardcoded string.

**Director runs before prompt assembly.**
The Director checks for three signals: semantic repetition (cosine > 0.85 via `all-MiniLM-L6-v2`), progress stall (>5 turns with no proposal-type keywords), and domain drift (low score + no domain vocabulary). When a hint fires, it is injected as a `[INTERNAL COACHING CUE]` block in the system prompt so the NPC delivers it naturally in its own voice. A 3-turn cooldown prevents hint spam.

**RAG is gated.**
Messages under 4 words skip embedding entirely. Messages over 4 words query the FAISS index; results are only injected if the top cosine score ≥ 0.55 to avoid polluting responses with irrelevant context.

---

## Running without an API key

The backend starts without `OPENAI_API_KEY`. Safety falls back to keyword-only detection; rapport Tier-2 and LLM calls are skipped. The NPC returns a placeholder message noting the key is missing. All other layers (RAG gating, Director signals, Tier-1 rapport, state tracking) remain active.

---

## Report

The full technical report — covering the rapport engine design, RAG pipeline, safety guard, Director layer, and persona-agnostic architecture — is at [`docs/docs.pdf`](docs/docs.pdf).
