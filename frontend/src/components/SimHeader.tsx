interface Props {
  personaName: string;
  onReset: () => void;
}

export function SimHeader({ personaName, onReset }: Props) {
  return (
    <header className="flex items-center justify-between border-b border-neutral-800 bg-neutral-950 px-6 py-4">
      <div>
        <h1 className="text-lg font-semibold text-white">
          AI Co-Worker — {personaName}
        </h1>
        <p className="text-xs text-neutral-400">
          Edtronaut · HRM Talent & Leadership · Gucci Global Group
        </p>
      </div>
      <button
        onClick={onReset}
        className="rounded-md border border-neutral-700 px-3 py-1.5 text-sm text-neutral-300 hover:bg-neutral-800"
      >
        Reset session
      </button>
    </header>
  );
}
