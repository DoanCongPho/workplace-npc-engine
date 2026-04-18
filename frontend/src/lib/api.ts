import type { ChatRequest, ChatResponse } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function postChat(req: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
  return res.json();
}

export async function resetSession(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/sessions/${sessionId}/reset`, { method: "POST" });
}
