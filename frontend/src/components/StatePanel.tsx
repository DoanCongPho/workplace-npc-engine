import type { StateUpdate } from "../types";

interface Props {
  state: StateUpdate;
  sessionId: string;
  personaId: string;
}

function rapportColor(score: number) {
  if (score >= 75) return "bg-emerald-500";
  if (score >= 50) return "bg-sky-500";
  if (score >= 30) return "bg-amber-500";
  return "bg-rose-500";
}

function tierBadge(tier: string) {
  const t = tier.toLowerCase();
  if (t === "engaged" || t === "energized")
    return "bg-emerald-100 text-emerald-700 border-emerald-200";
  if (t === "neutral" || t === "supportive")
    return "bg-sky-100 text-sky-700 border-sky-200";
  if (t === "guarded" || t === "concerned")
    return "bg-amber-100 text-amber-700 border-amber-200";
  return "bg-rose-100 text-rose-700 border-rose-200";
}

function MomentumArrow({ value }: { value: number }) {
  if (value > 1.5)
    return <span className="text-emerald-600 font-bold">↑</span>;
  if (value < -1.5)
    return <span className="text-rose-600 font-bold">↓</span>;
  return <span className="text-slate-400 font-bold">→</span>;
}

export function StatePanel({ state, sessionId }: Props) {
  const barColor = rapportColor(state.rapport_score);
  const badgeCls = tierBadge(state.emotional_state);

  return (
    <aside className="hidden w-72 flex-col gap-5 border-l border-slate-200 bg-white p-5 lg:flex">
      <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-400">
        Engine State
      </h2>

      {/* Rapport score */}
      <div className="space-y-2">
        <div className="flex items-baseline justify-between">
          <span className="text-xs font-medium text-slate-500">Rapport</span>
          <span className="text-2xl font-bold text-slate-800">
            {state.rapport_score.toFixed(0)}
            <span className="text-sm font-normal text-slate-400">/100</span>
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
          <div
            className={`h-full rounded-full transition-all duration-500 ${barColor}`}
            style={{ width: `${Math.min(100, Math.max(0, state.rapport_score))}%` }}
          />
        </div>
      </div>

      {/* Emotional tier badge */}
      <div className="space-y-1.5">
        <p className="text-xs font-medium text-slate-500">Current Tier</p>
        <span
          className={`inline-block rounded-md border px-2.5 py-1 text-xs font-semibold ${badgeCls}`}
        >
          {state.emotional_state}
        </span>
      </div>

      {/* Momentum */}
      <div className="space-y-1.5">
        <p className="text-xs font-medium text-slate-500">Momentum</p>
        <div className="flex items-center gap-1.5">
          <MomentumArrow value={state.rapport_momentum} />
          <span className="text-sm font-semibold text-slate-700">
            {state.rapport_momentum > 0 ? "+" : ""}
            {state.rapport_momentum.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Turn counter */}
      <div className="space-y-1.5">
        <p className="text-xs font-medium text-slate-500">Turn</p>
        <p className="text-sm font-semibold text-slate-700">{state.turn_count}</p>
      </div>

      <div className="mt-auto space-y-1 rounded-lg bg-slate-50 p-2.5">
        <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
          Session
        </p>
        <p className="break-all text-[10px] font-mono text-slate-500">{sessionId}</p>
      </div>
    </aside>
  );
}
