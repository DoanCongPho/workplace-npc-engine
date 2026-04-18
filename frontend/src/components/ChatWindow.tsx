import { useEffect, useRef, useState, type FormEvent } from "react";
import type { Message } from "../types";

interface Props {
  messages: Message[];
  isSending: boolean;
  error: string | null;
  onSend: (text: string) => void;
}

export function ChatWindow({ messages, isSending, error, onSend }: Props) {
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, isSending]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!draft.trim() || isSending) return;
    onSend(draft);
    setDraft("");
  }

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <p className="text-center text-sm text-neutral-500">
            Start a conversation with the AI co-worker.
          </p>
        )}
        <ul className="space-y-3">
          {messages.map((m, i) => (
            <li
              key={i}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[75%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm ${
                  m.role === "user"
                    ? "bg-indigo-600 text-white"
                    : "bg-neutral-800 text-neutral-100"
                }`}
              >
                {m.content}
              </div>
            </li>
          ))}
          {isSending && (
            <li className="flex justify-start">
              <div className="rounded-lg bg-neutral-800 px-3 py-2 text-sm text-neutral-400">
                …
              </div>
            </li>
          )}
        </ul>
        {error && (
          <p className="mt-3 text-center text-xs text-red-400">{error}</p>
        )}
      </div>

      <form
        onSubmit={handleSubmit}
        className="border-t border-neutral-800 bg-neutral-950 p-3"
      >
        <div className="flex gap-2">
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Type a message…"
            disabled={isSending}
            className="flex-1 rounded-md border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-white placeholder-neutral-500 focus:border-indigo-500 focus:outline-none disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isSending || !draft.trim()}
            className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-40"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
