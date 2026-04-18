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
    <div className="mx-4 mb-2 flex items-start gap-2 rounded-md border border-amber-700/50 bg-amber-900/30 px-3 py-2 text-sm text-amber-200">
      <span className="font-semibold">Hint</span>
      <p className="flex-1">{hint}</p>
      <button
        onClick={() => setDismissed(hint)}
        className="text-amber-300 hover:text-amber-100"
      >
        Dismiss
      </button>
    </div>
  );
}
