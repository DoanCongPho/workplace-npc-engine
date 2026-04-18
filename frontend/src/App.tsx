import { ChatWindow } from "./components/ChatWindow";
import { DirectorHintBanner } from "./components/DirectorHintBanner";
import { SimHeader } from "./components/SimHeader";
import { StatePanel } from "./components/StatePanel";
import { useConversation } from "./hooks/useConversation";

export default function App() {
  const { sessionId, messages, state, isSending, error, send, reset } =
    useConversation("ceo");

  return (
    <div className="flex h-full flex-col bg-neutral-900 text-white">
      <SimHeader personaName="Lorenzo Bertelli (CEO)" onReset={reset} />
      <div className="flex flex-1 overflow-hidden">
        <main className="flex flex-1 flex-col">
          <div className="pt-3">
            <DirectorHintBanner hint={state.director_hint} />
          </div>
          <div className="flex-1 overflow-hidden">
            <ChatWindow
              messages={messages}
              isSending={isSending}
              error={error}
              onSend={send}
            />
          </div>
        </main>
        <StatePanel state={state} sessionId={sessionId} />
      </div>
    </div>
  );
}
