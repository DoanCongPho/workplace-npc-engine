import { useState } from "react";
import { ChatWindow } from "./components/ChatWindow";
import { DirectorHintBanner } from "./components/DirectorHintBanner";
import { SimHeader } from "./components/SimHeader";
import { StatePanel } from "./components/StatePanel";
import { useConversation } from "./hooks/useConversation";
import { PERSONAS } from "./lib/personas";

export default function App() {
  const [personaId, setPersonaId] = useState("ceo");
  const { sessionId, messages, state, isSending, error, send, reset } =
    useConversation(personaId);

  const persona = PERSONAS[personaId];

  return (
    <div className="flex h-full flex-col bg-slate-50">
      <SimHeader
        personaId={personaId}
        onSwitchPersona={setPersonaId}
        onReset={reset}
      />
      <DirectorHintBanner hint={state.director_hint} />
      <div className="flex flex-1 overflow-hidden">
        <main className="flex flex-1 flex-col overflow-hidden">
          <ChatWindow
            messages={messages}
            isSending={isSending}
            error={error}
            onSend={send}
            personaId={personaId}
            personaInitials={persona.initials}
          />
        </main>
        <StatePanel state={state} sessionId={sessionId} personaId={personaId} />
      </div>
    </div>
  );
}
