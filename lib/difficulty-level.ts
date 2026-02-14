import { createContext, useContext } from "react";
import type { DifficultyFilter } from "./feed-content";

export interface DifficultyLevelContextValue {
  level: DifficultyFilter;
  setLevel: (level: DifficultyFilter) => void;
}

export const DifficultyLevelContext =
  createContext<DifficultyLevelContextValue | null>(null);

export function useDifficultyLevel(): DifficultyLevelContextValue {
  const ctx = useContext(DifficultyLevelContext);
  if (!ctx) {
    throw new Error(
      "useDifficultyLevel must be used within DifficultyLevelProvider",
    );
  }
  return ctx;
}
