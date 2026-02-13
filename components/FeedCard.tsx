import React from "react";
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Platform,
} from "react-native";
import { Ionicons, Feather } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import * as WebBrowser from "expo-web-browser";
import Colors from "@/constants/colors";
import { CodeBlock } from "@/components/CodeBlock";
import { useBookmarks } from "@/lib/bookmarks";
import type { FeedItem } from "@/lib/feed-content";

interface FeedCardProps {
  item: FeedItem;
}

export function FeedCard({ item }: FeedCardProps) {
  const { isBookmarked, toggleBookmark } = useBookmarks();
  const saved = isBookmarked(item.id);
  const catColor = Colors.categoryColors[item.category] || {
    bg: Colors.light.tint,
    text: "#FFF",
  };
  const diffColor = item.difficulty
    ? Colors.difficultyColors[item.difficulty]
    : null;

  const handleLink = async () => {
    if (Platform.OS !== "web") {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
    try {
      await WebBrowser.openBrowserAsync(item.sourceUrl);
    } catch {}
  };

  const handleBookmark = () => {
    if (Platform.OS !== "web") {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    }
    toggleBookmark(item);
  };

  return (
    <View style={styles.card}>
      <View style={styles.topRow}>
        <View style={[styles.categoryBadge, { backgroundColor: catColor.bg }]}>
          <Text style={[styles.categoryText, { color: catColor.text }]}>
            {item.category}
          </Text>
        </View>
        <View style={styles.topRight}>
          {diffColor ? (
            <View
              style={[
                styles.diffBadge,
                { backgroundColor: diffColor.bg },
              ]}
            >
              <Text style={[styles.diffText, { color: diffColor.text }]}>
                {item.difficulty}
              </Text>
            </View>
          ) : null}
          <Text style={styles.readTime}>
            {item.readTime}
          </Text>
        </View>
      </View>

      <Text style={styles.title}>{item.title}</Text>
      <Text style={styles.summary}>{item.summary}</Text>

      {item.codeSnippet ? (
        <CodeBlock code={item.codeSnippet} lang={item.codeLang} />
      ) : null}

      {item.tags.length > 0 ? (
        <View style={styles.tagsRow}>
          {item.tags.map((tag) => (
            <View key={tag} style={styles.tag}>
              <Text style={styles.tagText}>{tag}</Text>
            </View>
          ))}
        </View>
      ) : null}

      <View style={styles.footer}>
        <Pressable
          onPress={handleLink}
          style={({ pressed }) => [
            styles.sourceBtn,
            pressed && { opacity: 0.7 },
          ]}
        >
          <Feather name="external-link" size={14} color={Colors.light.tint} />
          <Text style={styles.sourceText} numberOfLines={1}>
            {item.source}
          </Text>
        </Pressable>
        <Pressable
          onPress={handleBookmark}
          style={({ pressed }) => [pressed && { opacity: 0.7 }]}
          hitSlop={12}
        >
          <Ionicons
            name={saved ? "bookmark" : "bookmark-outline"}
            size={22}
            color={saved ? Colors.light.tint : Colors.light.textSecondary}
          />
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: Colors.light.surface,
    marginHorizontal: 16,
    marginBottom: 12,
    borderRadius: 16,
    padding: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 3,
  },
  topRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 10,
  },
  categoryBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  categoryText: {
    fontSize: 11,
    fontFamily: "Inter_600SemiBold",
    letterSpacing: 0.3,
  },
  topRight: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  diffBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 10,
  },
  diffText: {
    fontSize: 10,
    fontFamily: "Inter_600SemiBold",
    letterSpacing: 0.2,
  },
  readTime: {
    fontSize: 11,
    color: Colors.light.textSecondary,
    fontFamily: "Inter_400Regular",
  },
  title: {
    fontSize: 17,
    fontFamily: "Inter_600SemiBold",
    color: Colors.light.text,
    lineHeight: 22,
    marginBottom: 6,
  },
  summary: {
    fontSize: 14,
    fontFamily: "Inter_400Regular",
    color: Colors.light.textSecondary,
    lineHeight: 20,
  },
  tagsRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
    marginTop: 12,
  },
  tag: {
    backgroundColor: Colors.light.surfaceSecondary,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: Colors.light.border,
  },
  tagText: {
    fontSize: 11,
    fontFamily: "Inter_400Regular",
    color: Colors.light.textSecondary,
  },
  footer: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 14,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: Colors.light.border,
  },
  sourceBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    flex: 1,
    marginRight: 12,
  },
  sourceText: {
    fontSize: 13,
    fontFamily: "Inter_400Regular",
    color: Colors.light.tint,
    flex: 1,
  },
});
