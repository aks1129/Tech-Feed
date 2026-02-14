# AIFeedX

## Overview

AIFeedX is a mobile-first educational feed application built with Expo (React Native) that delivers curated AI/tech learning content in a scrollable card format. Users can browse feed items across categories like Data Structures, System Design, ML/DL, GenAI, Agentic AI, and more. Each feed card includes a title, summary, optional code snippet with syntax highlighting, difficulty level, source link, and bookmarking functionality. The app has two main tabs: a Feed tab with category/difficulty filtering and infinite scroll, and a Saved tab for bookmarked items. Content is currently hardcoded on the client side (in `lib/feed-content.ts`), with a backend server scaffolded but not yet serving feed data.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend (Expo / React Native)

- **Framework**: Expo SDK 54 with expo-router v6 for file-based routing
- **Navigation**: Tab-based layout using expo-router's `Tabs` component with two screens: Feed (`index.tsx`) and Saved (`saved.tsx`). Supports native tabs via `expo-router/unstable-native-tabs` when liquid glass is available (iOS 26+), otherwise falls back to classic tab layout with blur effects.
- **State Management**: React Context for bookmarks (`BookmarkProvider`) and difficulty level (`DifficultyLevelProvider`), both persisted to AsyncStorage. TanStack React Query is set up (`lib/query-client.ts`) for server data fetching but currently unused since feed content is client-side.
- **Feed Content**: All feed items are defined as static arrays in `lib/feed-content.ts` with a `getFeedPage()` function for paginated access with category and difficulty filtering. This is the data layer that would eventually be replaced by API calls.
- **UI Components**:
  - `FeedCard` - Main content card with category badge, difficulty badge, code snippet, bookmark toggle, and external link opening
  - `CategoryChips` - Horizontal scrollable category filter
  - `DifficultySelector` - Difficulty level filter (All/Easy/Medium/Hard)
  - `CodeBlock` - Syntax-highlighted code display with keyword/string/comment/function coloring
- **Fonts**: Inter (400, 600) via `@expo-google-fonts/inter`
- **Haptics**: Used throughout for tactile feedback on interactions (category selection, bookmarking, etc.)
- **Platform Support**: iOS, Android, and Web. Platform-specific adaptations throughout (haptics disabled on web, different tab bar styles, keyboard handling)

### Backend (Express)

- **Framework**: Express 5 with TypeScript, compiled via `tsx` for development and `esbuild` for production
- **Current State**: Minimal scaffolding with CORS setup, static file serving for production builds, and a landing page template. No API routes are implemented yet — `server/routes.ts` is empty.
- **Storage**: `server/storage.ts` defines an `IStorage` interface and `MemStorage` in-memory implementation with basic user CRUD. Not connected to any routes.
- **CORS**: Dynamically configured to allow Replit domains and localhost origins for Expo web development.
- **Production**: Serves static Expo web build from `dist/` directory. Development mode proxies to Metro bundler.

### Database

- **ORM**: Drizzle ORM configured for PostgreSQL (`drizzle.config.ts`)
- **Schema**: Single `users` table in `shared/schema.ts` with id (UUID), username, and password fields. Validation via `drizzle-zod`.
- **Current State**: Schema exists but the database is not actively used. The `MemStorage` class is the active storage backend. Run `npm run db:push` to push schema to Postgres.

### Build & Deployment

- **Development**: Two processes — `expo:dev` for Metro bundler and `server:dev` for Express server
- **Production Build**: `expo:static:build` runs a custom build script (`scripts/build.js`) that starts Metro, fetches the bundle, and saves static assets to `dist/`. Server is built with esbuild to `server_dist/`.
- **Production Run**: `server:prod` serves the built static files and API from a single Express server

### Path Aliases

- `@/*` maps to project root
- `@shared/*` maps to `./shared/*`

## External Dependencies

- **PostgreSQL**: Configured via `DATABASE_URL` environment variable, used with Drizzle ORM. Schema defined but not actively queried yet.
- **AsyncStorage** (`@react-native-async-storage/async-storage`): Client-side persistence for bookmarks and difficulty preferences.
- **TanStack React Query**: Set up for API data fetching with `EXPO_PUBLIC_DOMAIN` environment variable for API URL resolution. Currently scaffolded but not actively used.
- **Expo Services**: expo-web-browser (opening source URLs), expo-haptics (tactile feedback), expo-image-picker, expo-location (imported but usage not visible in provided files), expo-splash-screen, expo-font.
- **Environment Variables**:
  - `DATABASE_URL` — PostgreSQL connection string (required for db:push)
  - `EXPO_PUBLIC_DOMAIN` — API server domain for client-server communication
  - `REPLIT_DEV_DOMAIN` — Replit development domain (auto-set by Replit)
  - `REPLIT_DOMAINS` — Comma-separated Replit domains for CORS
  - `REPLIT_INTERNAL_APP_DOMAIN` — Replit deployment domain