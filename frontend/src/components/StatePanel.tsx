import type { StateUpdate } from "../types";

interface Props {
  state: StateUpdate;
  sessionId: string;
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-neutral-500">{label}</p>
      <p className="mt-1 text-sm text-neutral-100">{value}</p>
    </div>
  );
}

export function StatePanel({ state, sessionId }: Props) {
  return (
    <aside className="hidden w-72 flex-col gap-4 border-l border-neutral-800 bg-neutral-950 p-4 lg:flex">
      <h2 className="text-sm font-semibold text-white">NPC State</h2>
      <Stat label="Rapport score" value={state.rapport_score.toFixed(1)} />
      <Stat label="Momentum" value={state.rapport_momentum.toFixed(2)} />
      <Stat label="Emotion" value={state.emotional_state} />
      <Stat label="Turn" value={state.turn_count} />
      <div className="mt-auto break-all text-[10px] text-neutral-600">
        session: {sessionId}
      </div>
    </aside>
  );
}
