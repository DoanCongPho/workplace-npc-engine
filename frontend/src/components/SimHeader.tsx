import { PERSONAS } from "../lib/personas";

interface Props {
  personaId: string;
  onSwitchPersona: (id: string) => void;
  onReset: () => void;
}

export function SimHeader({ personaId, onSwitchPersona, onReset }: Props) {
  const persona = PERSONAS[personaId];

  return (
    <header className="border-b border-slate-200 bg-white px-6 py-3 shadow-sm">
      <div className="flex items-center justify-between gap-4">
        {/* Left: branding */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-indigo-600 text-xs font-bold text-white">
            E
          </div>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-slate-800">Edtronaut</p>
            <p className="text-xs text-slate-400 truncate">HRM · Gucci Group Simulation</p>
          </div>
        </div>

        {/* Right: switcher + active persona + reset */}
        <div className="flex items-center gap-3 flex-shrink-0">
          {/* Persona tabs */}
          <div className="flex items-center gap-1 rounded-xl bg-slate-100 p-1">
            {Object.values(PERSONAS).map((p) => {
              const isActive = p.id === personaId;
              const activeCls =
                p.id === "ceo"
                  ? "bg-white text-indigo-600 shadow-sm"
                  : "bg-white text-teal-600 shadow-sm";
              return (
                <button
                  key={p.id}
                  onClick={() => onSwitchPersona(p.id)}
                  className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                    isActive ? activeCls : "text-slate-400 hover:text-slate-600"
                  }`}
                >
                  {p.id.toUpperCase()}
                </button>
              );
            })}
          </div>

          {/* Active persona badge */}
          <div
            className={`hidden sm:flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs ${
              personaId === "ceo"
                ? "bg-indigo-50 border border-indigo-100"
                : "bg-teal-50 border border-teal-100"
            }`}
          >
            <div
              className={`flex h-6 w-6 items-center justify-center rounded-full text-[10px] font-bold ${
                personaId === "ceo"
                  ? "bg-indigo-200 text-indigo-700"
                  : "bg-teal-200 text-teal-700"
              }`}
            >
              {persona.initials}
            </div>
            <div>
              <p
                className={`font-semibold leading-none ${
                  personaId === "ceo" ? "text-indigo-700" : "text-teal-700"
                }`}
              >
                {persona.name}
              </p>
              <p className="text-slate-400 leading-none mt-0.5">{persona.title.split("·")[0].trim()}</p>
            </div>
          </div>

          <button
            onClick={onReset}
            className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-500 hover:border-slate-300 hover:text-slate-700 transition-colors"
          >
            Reset
          </button>
        </div>
      </div>
    </header>
  );
}
