import React, { useState, useEffect, useMemo, ReactNode, useCallback } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { BookmarkContext, getOriginalId } from "@/lib/bookmarks";
import type { FeedItem } from "@/lib/feed-content";

const STORAGE_KEY = "techpulse_bookmarks";

export function BookmarkProvider({ children }: { children: ReactNode }) {
  const [bookmarks, setBookmarks] = useState<FeedItem[]>([]);

  useEffect(() => {
    AsyncStorage.getItem(STORAGE_KEY).then((raw) => {
      if (raw) {
        try {
          setBookmarks(JSON.parse(raw));
        } catch {}
      }
    });
  }, []);

  const persist = useCallback((items: FeedItem[]) => {
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(items)).catch(() => {});
  }, []);

  const isBookmarked = useCallback(
    (id: string) => {
      const origId = getOriginalId(id);
      return bookmarks.some((b) => getOriginalId(b.id) === origId);
    },
    [bookmarks],
  );

  const toggleBookmark = useCallback(
    (item: FeedItem) => {
      const origId = getOriginalId(item.id);
      const exists = bookmarks.some((b) => getOriginalId(b.id) === origId);
      let updated: FeedItem[];
      if (exists) {
        updated = bookmarks.filter((b) => getOriginalId(b.id) !== origId);
      } else {
        updated = [item, ...bookmarks];
      }
      setBookmarks(updated);
      persist(updated);
    },
    [bookmarks, persist],
  );

  const value = useMemo(
    () => ({ bookmarks, isBookmarked, toggleBookmark }),
    [bookmarks, isBookmarked, toggleBookmark],
  );

  return (
    <BookmarkContext.Provider value={value}>
      {children}
    </BookmarkContext.Provider>
  );
}
