import { createContext, useContext } from "react";
import type { FeedItem } from "./feed-content";

export interface BookmarkContextValue {
  bookmarks: FeedItem[];
  isBookmarked: (originalId: string) => boolean;
  toggleBookmark: (item: FeedItem) => void;
}

export const BookmarkContext = createContext<BookmarkContextValue | null>(null);

export function useBookmarks(): BookmarkContextValue {
  const ctx = useContext(BookmarkContext);
  if (!ctx) {
    throw new Error("useBookmarks must be used within BookmarkProvider");
  }
  return ctx;
}

export function getOriginalId(id: string): string {
  return id.replace(/-page\d+-\w+$/, "");
}
