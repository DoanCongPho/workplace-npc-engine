import { useCallback, useEffect, useRef, useState } from "react";
import { postChat, resetSession } from "../lib/api";
import type { Message, StateUpdate } from "../types";

const INITIAL_STATE: StateUpdate = {
  rapport_score: 50,
  rapport_momentum: 0,
  emotional_state: "Neutral",
  turn_count: 0,
  director_hint: null,
};

function newSessionId(): string {
  return `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export function useConversation(personaId: string = "ceo") {
  const [sessionId, setSessionId] = useState<string>(() => newSessionId());
  const [messages, setMessages] = useState<Message[]>([]);
  const [state, setState] = useState<StateUpdate>(INITIAL_STATE);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset conversation when persona switches (skip on first mount)
  const firstRender = useRef(true);
  useEffect(() => {
    if (firstRender.current) { firstRender.current = false; return; }
    setSessionId(newSessionId());
    setMessages([]);
    setState(INITIAL_STATE);
    setError(null);
  }, [personaId]);

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || isSending) return;
      setError(null);
      setIsSending(true);
      setMessages((m) => [...m, { role: "user", content: trimmed }]);
      try {
        const res = await postChat({
          persona_id: personaId,
          session_id: sessionId,
          user_message: trimmed,
        });
        setMessages((m) => [
          ...m,
          { role: "assistant", content: res.assistant_message },
        ]);
        setState(res.state);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setIsSending(false);
      }
    },
    [personaId, sessionId, isSending],
  );

  const reset = useCallback(async () => {
    await resetSession(sessionId);
    setSessionId(newSessionId());
    setMessages([]);
    setState(INITIAL_STATE);
    setError(null);
  }, [sessionId]);

  return { sessionId, messages, state, isSending, error, send, reset };
}
