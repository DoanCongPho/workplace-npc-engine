import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import type { Message } from "../types";

interface Props {
  messages: Message[];
  isSending: boolean;
  error: string | null;
  onSend: (text: string) => void;
  personaId: string;
  personaInitials: string;
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1 px-1 py-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-2 w-2 rounded-full bg-slate-400 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  );
}

export function ChatWindow({
  messages,
  isSending,
  error,
  onSend,
  personaId,
  personaInitials,
}: Props) {
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, isSending]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    submit();
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    if (!draft.trim() || isSending) return;
    onSend(draft);
    setDraft("");
  }

  // Persona-specific accent classes (full strings so Tailwind doesn't purge)
  const userBubbleCls =
    personaId === "ceo"
      ? "bg-indigo-500 text-white"
      : "bg-teal-500 text-white";

  const avatarCls =
    personaId === "ceo"
      ? "bg-indigo-100 text-indigo-700"
      : "bg-teal-100 text-teal-700";

  const sendBtnCls =
    personaId === "ceo"
      ? "bg-indigo-500 hover:bg-indigo-600 disabled:bg-indigo-300"
      : "bg-teal-500 hover:bg-teal-600 disabled:bg-teal-300";

  const focusRingCls =
    personaId === "ceo"
      ? "focus:ring-indigo-400"
      : "focus:ring-teal-400";

  return (
    <div className="flex h-full flex-col bg-slate-50">
      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 gap-3">
            <div
              className={`flex h-12 w-12 items-center justify-center rounded-full text-lg font-bold ${avatarCls}`}
            >
              {personaInitials}
            </div>
            <p className="text-sm text-slate-400">
              Start a conversation to begin the simulation.
            </p>
          </div>
        ) : (
          <ul className="space-y-4">
            {messages.map((m, i) => (
              <li
                key={i}
                className={`flex items-end gap-2 ${
                  m.role === "user" ? "flex-row-reverse" : "flex-row"
                }`}
              >
                {/* Avatar — only for assistant */}
                {m.role === "assistant" && (
                  <div
                    className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full text-xs font-semibold ${avatarCls}`}
                  >
                    {personaInitials}
                  </div>
                )}

                <div
                  className={`max-w-[72%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed shadow-sm ${
                    m.role === "user"
                      ? `${userBubbleCls} rounded-br-sm`
                      : "bg-white text-slate-800 border border-slate-100 rounded-bl-sm"
                  }`}
                >
                  {m.content}
                </div>
              </li>
            ))}

            {isSending && (
              <li className="flex items-end gap-2 flex-row">
                <div
                  className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full text-xs font-semibold ${avatarCls}`}
                >
                  {personaInitials}
                </div>
                <div className="rounded-2xl rounded-bl-sm border border-slate-100 bg-white px-4 py-2.5 shadow-sm">
                  <TypingDots />
                </div>
              </li>
            )}
          </ul>
        )}

        {error && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-center text-xs text-red-600">
            {error}
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-slate-200 bg-white px-4 py-3">
        <form onSubmit={handleSubmit} className="flex items-end gap-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
            disabled={isSending}
            rows={1}
            className={`flex-1 resize-none rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 ${focusRingCls} disabled:opacity-50`}
            style={{ maxHeight: "120px" }}
            onInput={(e) => {
              const t = e.currentTarget;
              t.style.height = "auto";
              t.style.height = `${Math.min(t.scrollHeight, 120)}px`;
            }}
          />
          <button
            type="submit"
            disabled={isSending || !draft.trim()}
            className={`rounded-xl px-4 py-2.5 text-sm font-medium text-white transition-colors ${sendBtnCls} disabled:cursor-not-allowed`}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
