import React from "react";
import {
  View,
  Pressable,
  Text,
  StyleSheet,
  Platform,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import Colors from "@/constants/colors";
import { ALL_DIFFICULTIES, type DifficultyFilter } from "@/lib/feed-content";
import { useDifficultyLevel } from "@/lib/difficulty-level";

const levelIcons: Record<string, string> = {
  All: "shuffle",
  Easy: "flash-outline",
  Medium: "trending-up-outline",
  Hard: "rocket-outline",
};

const levelDescriptions: Record<string, string> = {
  All: "Random mix",
  Easy: "Fundamentals",
  Medium: "Intermediate",
  Hard: "Advanced",
};

export function DifficultySelector() {
  const { level, setLevel } = useDifficultyLevel();

  const handlePress = (diff: DifficultyFilter) => {
    if (Platform.OS !== "web") {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
    setLevel(diff);
  };

  return (
    <View style={styles.container}>
      {ALL_DIFFICULTIES.map((diff) => {
        const isSelected = diff === level;
        const color =
          diff === "All"
            ? { bg: Colors.light.tint, text: "#FFF" }
            : Colors.difficultyColors[diff] || { bg: Colors.light.tint, text: "#FFF" };

        return (
          <Pressable
            key={diff}
            onPress={() => handlePress(diff)}
            style={({ pressed }) => [
              styles.pill,
              isSelected
                ? { backgroundColor: color.bg }
                : {
                    backgroundColor: Colors.light.surface,
                    borderWidth: 1,
                    borderColor: Colors.light.border,
                  },
              pressed && { opacity: 0.8, transform: [{ scale: 0.96 }] },
            ]}
          >
            <Ionicons
              name={levelIcons[diff] as any}
              size={14}
              color={isSelected ? color.text : Colors.light.textSecondary}
            />
            <Text
              style={[
                styles.pillText,
                isSelected
                  ? { color: color.text, fontFamily: "Inter_600SemiBold" }
                  : { color: Colors.light.textSecondary, fontFamily: "Inter_400Regular" },
              ]}
            >
              {levelDescriptions[diff]}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    paddingHorizontal: 16,
    paddingVertical: 8,
    gap: 6,
    backgroundColor: Colors.light.surface,
    borderBottomWidth: 1,
    borderBottomColor: Colors.light.border,
  },
  pill: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 4,
    paddingVertical: 8,
    borderRadius: 20,
  },
  pillText: {
    fontSize: 11,
    lineHeight: 14,
  },
});
