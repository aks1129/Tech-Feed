import React, { useState, useEffect, useMemo, ReactNode, useCallback } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { DifficultyLevelContext } from "@/lib/difficulty-level";
import type { DifficultyFilter } from "@/lib/feed-content";

const STORAGE_KEY = "aifeedx_difficulty_level";

export function DifficultyLevelProvider({ children }: { children: ReactNode }) {
  const [level, setLevelState] = useState<DifficultyFilter>("All");

  useEffect(() => {
    AsyncStorage.getItem(STORAGE_KEY).then((raw) => {
      if (raw && (raw === "All" || raw === "Easy" || raw === "Medium" || raw === "Hard")) {
        setLevelState(raw as DifficultyFilter);
      }
    });
  }, []);

  const setLevel = useCallback((newLevel: DifficultyFilter) => {
    setLevelState(newLevel);
    AsyncStorage.setItem(STORAGE_KEY, newLevel).catch(() => {});
  }, []);

  const value = useMemo(() => ({ level, setLevel }), [level, setLevel]);

  return (
    <DifficultyLevelContext.Provider value={value}>
      {children}
    </DifficultyLevelContext.Provider>
  );
}
