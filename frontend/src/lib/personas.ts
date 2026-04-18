export interface PersonaInfo {
  id: string;
  name: string;
  title: string;
  initials: string;
}

export const PERSONAS: Record<string, PersonaInfo> = {
  ceo: {
    id: "ceo",
    name: "Lorenzo Bertelli",
    title: "Group CEO · Gucci Group",
    initials: "LB",
  },
  chro: {
    id: "chro",
    name: "Group CHRO",
    title: "Chief Human Resources Officer · Gucci Group",
    initials: "HR",
  },
};
