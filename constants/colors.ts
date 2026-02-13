const accent = "#0A84FF";
const accentDim = "rgba(10, 132, 255, 0.12)";

const categoryColors: Record<string, { bg: string; text: string }> = {
  "Data Structures": { bg: "#FF6B35", text: "#FFFFFF" },
  "System Design": { bg: "#7B2FF7", text: "#FFFFFF" },
  "ML / DL": { bg: "#00C9A7", text: "#FFFFFF" },
  "GenAI": { bg: "#FF3CAC", text: "#FFFFFF" },
  "Agentic AI": { bg: "#F5A623", text: "#1A1A1A" },
  "AIOps": { bg: "#4ECDC4", text: "#1A1A1A" },
  "Agentic Ops": { bg: "#E63946", text: "#FFFFFF" },
  "Deployment": { bg: "#2EC4B6", text: "#FFFFFF" },
  "Tech Stacks": { bg: "#845EC2", text: "#FFFFFF" },
};

const difficultyColors: Record<string, { bg: string; text: string }> = {
  "Easy": { bg: "#34C759", text: "#FFFFFF" },
  "Medium": { bg: "#FF9500", text: "#FFFFFF" },
  "Hard": { bg: "#FF3B30", text: "#FFFFFF" },
};

export default {
  light: {
    text: "#1A1A2E",
    textSecondary: "#6B7280",
    background: "#F0F2F5",
    surface: "#FFFFFF",
    surfaceSecondary: "#F7F8FA",
    border: "#E5E7EB",
    tint: accent,
    tintDim: accentDim,
    tabIconDefault: "#9CA3AF",
    tabIconSelected: accent,
    codeBackground: "#1E1E2E",
    codeText: "#CDD6F4",
    codeKeyword: "#CBA6F7",
    codeString: "#A6E3A1",
    codeComment: "#6C7086",
    codeFunction: "#89B4FA",
    codeNumber: "#FAB387",
  },
  categoryColors,
  difficultyColors,
};
