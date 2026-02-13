import React from "react";
import {
  ScrollView,
  Pressable,
  Text,
  StyleSheet,
  Platform,
} from "react-native";
import * as Haptics from "expo-haptics";
import Colors from "@/constants/colors";
import { ALL_CATEGORIES, type Category } from "@/lib/feed-content";

interface CategoryChipsProps {
  selected: Category;
  onSelect: (cat: Category) => void;
}

export function CategoryChips({ selected, onSelect }: CategoryChipsProps) {
  const handlePress = (cat: Category) => {
    if (Platform.OS !== "web") {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
    onSelect(cat);
  };

  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.container}
      style={styles.scroll}
    >
      {ALL_CATEGORIES.map((cat) => {
        const isSelected = cat === selected;
        const catColor =
          cat === "All"
            ? { bg: Colors.light.tint, text: "#FFF" }
            : Colors.categoryColors[cat] || {
                bg: Colors.light.tint,
                text: "#FFF",
              };

        return (
          <Pressable
            key={cat}
            onPress={() => handlePress(cat)}
            style={({ pressed }) => [
              styles.chip,
              isSelected
                ? { backgroundColor: catColor.bg }
                : { backgroundColor: Colors.light.surface, borderWidth: 1, borderColor: Colors.light.border },
              pressed && { opacity: 0.8, transform: [{ scale: 0.96 }] },
            ]}
          >
            <Text
              style={[
                styles.chipText,
                isSelected
                  ? { color: catColor.text, fontFamily: "Inter_600SemiBold" }
                  : { color: Colors.light.textSecondary, fontFamily: "Inter_400Regular" },
              ]}
              numberOfLines={1}
            >
              {cat}
            </Text>
          </Pressable>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 0,
  },
  container: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 8,
    flexDirection: "row",
  },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
  },
  chipText: {
    fontSize: 13,
    lineHeight: 16,
  },
});
