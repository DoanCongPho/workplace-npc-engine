import { useState, useEffect } from "react";

interface Props {
  hint: string | null;
}

export function DirectorHintBanner({ hint }: Props) {
  const [dismissed, setDismissed] = useState<string | null>(null);

  useEffect(() => {
    setDismissed(null);
  }, [hint]);

  if (!hint || dismissed === hint) return null;

  return (
    <div className="mx-4 mt-3 flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 shadow-sm">
      <span className="text-base" aria-hidden>💡</span>
      <div className="flex-1">
        <p className="text-xs font-semibold uppercase tracking-wide text-amber-700 mb-0.5">
          Director Hint
        </p>
        <p className="text-sm text-amber-800">{hint}</p>
      </div>
      <button
        onClick={() => setDismissed(hint)}
        className="mt-0.5 rounded-md px-2 py-0.5 text-xs text-amber-600 hover:bg-amber-100 hover:text-amber-800 transition-colors"
      >
        Dismiss
      </button>
    </div>
  );
}
