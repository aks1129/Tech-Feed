import React, { useState, useCallback, useRef, useEffect } from "react";
import {
  View,
  FlatList,
  StyleSheet,
  RefreshControl,
  ActivityIndicator,
  Text,
  Platform,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { StatusBar } from "expo-status-bar";
import Colors from "@/constants/colors";
import { CategoryChips } from "@/components/CategoryChips";
import { DifficultySelector } from "@/components/DifficultySelector";
import { FeedCard } from "@/components/FeedCard";
import { useDifficultyLevel } from "@/lib/difficulty-level";
import { getFeedPage, type Category, type FeedItem } from "@/lib/feed-content";

const PAGE_SIZE = 5;

export default function FeedScreen() {
  const insets = useSafeAreaInsets();
  const { level } = useDifficultyLevel();
  const [category, setCategory] = useState<Category>("All");
  const [items, setItems] = useState<FeedItem[]>(() =>
    getFeedPage(0, PAGE_SIZE, "All", "All"),
  );
  const [page, setPage] = useState(1);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  useEffect(() => {
    setPage(1);
    setItems(getFeedPage(0, PAGE_SIZE, category, level));
    flatListRef.current?.scrollToOffset({ offset: 0, animated: false });
  }, [level]);

  const handleCategoryChange = useCallback(
    (cat: Category) => {
      setCategory(cat);
      setPage(1);
      setItems(getFeedPage(0, PAGE_SIZE, cat, level));
      flatListRef.current?.scrollToOffset({ offset: 0, animated: true });
    },
    [level],
  );

  const handleRefresh = useCallback(() => {
    setRefreshing(true);
    setTimeout(() => {
      setPage(1);
      setItems(getFeedPage(0, PAGE_SIZE, category, level));
      setRefreshing(false);
    }, 600);
  }, [category, level]);

  const handleLoadMore = useCallback(() => {
    if (loadingMore) return;
    setLoadingMore(true);
    setTimeout(() => {
      const newItems = getFeedPage(page, PAGE_SIZE, category, level);
      setItems((prev) => [...prev, ...newItems]);
      setPage((p) => p + 1);
      setLoadingMore(false);
    }, 400);
  }, [page, category, level, loadingMore]);

  const renderItem = useCallback(
    ({ item }: { item: FeedItem }) => <FeedCard item={item} />,
    [],
  );

  const keyExtractor = useCallback((item: FeedItem) => item.id, []);

  const webTopInset = Platform.OS === "web" ? 67 : 0;

  return (
    <View style={styles.container}>
      <StatusBar style="dark" />
      <View
        style={[
          styles.header,
          { paddingTop: (Platform.OS === "web" ? webTopInset : insets.top) + 8 },
        ]}
      >
        <Text style={styles.headerTitle}>AIFeedX</Text>
        <Text style={styles.headerSubtitle}>Principal AI Engineer Feed</Text>
      </View>
      <DifficultySelector />
      <CategoryChips selected={category} onSelect={handleCategoryChange} />
      <FlatList
        ref={flatListRef}
        data={items}
        renderItem={renderItem}
        keyExtractor={keyExtractor}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        onEndReached={handleLoadMore}
        onEndReachedThreshold={0.5}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            tintColor={Colors.light.tint}
            colors={[Colors.light.tint]}
          />
        }
        ListFooterComponent={
          loadingMore ? (
            <View style={styles.loadingFooter}>
              <ActivityIndicator size="small" color={Colors.light.tint} />
            </View>
          ) : null
        }
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <Text style={styles.emptyTitle}>No content found</Text>
            <Text style={styles.emptyText}>
              Try a different difficulty level or category
            </Text>
          </View>
        }
        initialNumToRender={5}
        maxToRenderPerBatch={5}
        windowSize={7}
        removeClippedSubviews={Platform.OS !== "web"}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.light.background,
  },
  header: {
    paddingHorizontal: 20,
    paddingBottom: 4,
    backgroundColor: Colors.light.surface,
  },
  headerTitle: {
    fontSize: 28,
    fontFamily: "Inter_600SemiBold",
    color: Colors.light.text,
    letterSpacing: -0.5,
  },
  headerSubtitle: {
    fontSize: 13,
    fontFamily: "Inter_400Regular",
    color: Colors.light.textSecondary,
    marginTop: 2,
  },
  listContent: {
    paddingTop: 8,
    paddingBottom: Platform.OS === "web" ? 34 : 100,
  },
  loadingFooter: {
    paddingVertical: 24,
    alignItems: "center",
  },
  emptyState: {
    paddingVertical: 60,
    alignItems: "center",
    gap: 8,
  },
  emptyTitle: {
    fontSize: 17,
    fontFamily: "Inter_600SemiBold",
    color: Colors.light.text,
  },
  emptyText: {
    fontSize: 14,
    fontFamily: "Inter_400Regular",
    color: Colors.light.textSecondary,
  },
});
