# AIFeedX

A mobile infinite-scrolling feed app built for principal-level AI engineers, delivering curated technical content across 9 categories.

## Categories

- **Data Structures** — 150 common interview questions and patterns
- **System Design** — Large-scale architecture and distributed systems
- **ML / DL** — Machine learning and deep learning fundamentals
- **GenAI** — Generative AI, RAG pipelines, prompt engineering
- **Agentic AI** — Agent frameworks, multi-agent orchestration, tool calling
- **AIOps** — ML model monitoring, feature stores, incident response
- **Agentic Ops** — LLM observability, guardrails, cost optimization
- **Deployment & Production** — CI/CD, Kubernetes, model serving
- **Tech Stacks** — Platform comparisons, vector databases, GPU providers

## Features

- **Infinite scrolling feed** with pull-to-refresh
- **Difficulty filtering** — Random mix, Fundamentals (Easy), Intermediate (Medium), Advanced (Hard)
- **Category filtering** — Browse by topic or view all
- **Bookmark system** — Save and revisit favorite items in the Saved tab
- **Code snippets** — Python syntax highlighting with a Catppuccin-inspired theme
- **Offline-ready** — All content and preferences stored locally on device
- **Persistent settings** — Difficulty level and bookmarks saved across sessions

## Tech Stack

- **Frontend:** React Native with Expo (Expo Router for navigation)
- **Backend:** Express + TypeScript (serves API and landing page)
- **State Management:** React Context + React Query
- **Local Storage:** AsyncStorage for bookmarks and preferences
- **Fonts:** Inter (Regular & SemiBold)
- **Icons:** @expo/vector-icons (Ionicons, Feather)

## Getting Started

1. Install dependencies:
   ```
   npm install
   ```

2. Start the backend server:
   ```
   npm run server:dev
   ```

3. Start the Expo dev server:
   ```
   npm run expo:dev
   ```

4. Open the app:
   - **Mobile:** Scan the QR code with Expo Go
   - **Web:** Visit `http://localhost:8081`

## Project Structure

```
app/                    # Expo Router screens
  _layout.tsx           # Root layout with providers
  (tabs)/
    _layout.tsx         # Tab navigation (Feed + Saved)
    index.tsx           # Feed screen
    saved.tsx           # Bookmarked items screen
components/             # Reusable UI components
  FeedCard.tsx          # Content card with code snippets
  CategoryChips.tsx     # Horizontal category filter
  DifficultySelector.tsx # Difficulty level pills
  BookmarkProvider.tsx  # Bookmark context provider
  DifficultyLevelProvider.tsx # Difficulty context provider
  ErrorBoundary.tsx     # Error boundary wrapper
lib/                    # Core logic and data
  feed-content.ts       # All feed content and pagination
  bookmarks.ts          # Bookmark context definition
  difficulty-level.ts   # Difficulty context definition
  query-client.ts       # React Query client setup
constants/
  colors.ts             # Color palette and theme
server/
  index.ts              # Express backend entry point
```
