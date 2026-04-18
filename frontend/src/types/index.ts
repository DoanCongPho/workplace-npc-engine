export type Role = "user" | "assistant";

export interface Message {
  role: Role;
  content: string;
}

export interface StateUpdate {
  rapport_score: number;
  rapport_momentum: number;
  emotional_state: string;
  turn_count: number;
  director_hint: string | null;
}

export interface ChatResponse {
  assistant_message: string;
  state: StateUpdate;
  safety_flags: Record<string, unknown>;
}

export interface ChatRequest {
  persona_id: string;
  session_id: string;
  user_message: string;
}
