import React from "react";
import { View, Text, StyleSheet, Platform } from "react-native";
import Colors from "@/constants/colors";

interface CodeBlockProps {
  code: string;
  lang?: string;
}

const monoFont = Platform.select({
  ios: "Menlo",
  android: "monospace",
  default: "monospace",
}) as string;

function highlightLine(line: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  let remaining = line;
  let keyIdx = 0;

  const keywords =
    /\b(def|class|return|if|else|elif|for|while|import|from|in|not|and|or|is|None|True|False|self|try|except|raise|with|as|lambda|yield|async|await|break|continue|pass|global|nonlocal|del|assert)\b/g;
  const commentRegex = /#.*/g;
  const stringRegex = /(["'`])(?:(?!\1|\\).|\\.)*?\1|f?"""[\s\S]*?"""|f?'''[\s\S]*?'''/g;
  const numberRegex = /\b\d+\.?\d*\b/g;
  const funcRegex = /\b([a-zA-Z_]\w*)\s*(?=\()/g;
  const decoratorRegex = /@\w+/g;

  const commentMatch = remaining.match(/^(\s*)(#.*)$/);
  if (commentMatch) {
    if (commentMatch[1]) {
      parts.push(
        <Text key={`ws-${keyIdx++}`} style={{ color: Colors.light.codeText }}>
          {commentMatch[1]}
        </Text>,
      );
    }
    parts.push(
      <Text key={`cm-${keyIdx++}`} style={{ color: Colors.light.codeComment }}>
        {commentMatch[2]}
      </Text>,
    );
    return parts;
  }

  let lastIndex = 0;
  const tokens: { start: number; end: number; color: string }[] = [];

  const applyRegex = (regex: RegExp, color: string) => {
    regex.lastIndex = 0;
    let m;
    while ((m = regex.exec(remaining)) !== null) {
      const overlap = tokens.some(
        (t) => m!.index < t.end && m!.index + m![0].length > t.start,
      );
      if (!overlap) {
        tokens.push({
          start: m.index,
          end: m.index + m[0].length,
          color,
        });
      }
    }
  };

  applyRegex(stringRegex, Colors.light.codeString);
  applyRegex(keywords, Colors.light.codeKeyword);
  applyRegex(decoratorRegex, Colors.light.codeKeyword);
  applyRegex(numberRegex, Colors.light.codeNumber);
  applyRegex(funcRegex, Colors.light.codeFunction);

  tokens.sort((a, b) => a.start - b.start);

  for (const token of tokens) {
    if (token.start > lastIndex) {
      parts.push(
        <Text key={`t-${keyIdx++}`} style={{ color: Colors.light.codeText }}>
          {remaining.slice(lastIndex, token.start)}
        </Text>,
      );
    }
    parts.push(
      <Text key={`h-${keyIdx++}`} style={{ color: token.color }}>
        {remaining.slice(token.start, token.end)}
      </Text>,
    );
    lastIndex = token.end;
  }

  if (lastIndex < remaining.length) {
    parts.push(
      <Text key={`e-${keyIdx++}`} style={{ color: Colors.light.codeText }}>
        {remaining.slice(lastIndex)}
      </Text>,
    );
  }

  return parts.length > 0
    ? parts
    : [
        <Text key="plain" style={{ color: Colors.light.codeText }}>
          {remaining}
        </Text>,
      ];
}

export function CodeBlock({ code, lang }: CodeBlockProps) {
  const lines = code.split("\n");

  return (
    <View style={styles.container}>
      {lang ? (
        <View style={styles.langBadge}>
          <Text style={styles.langText}>{lang}</Text>
        </View>
      ) : null}
      <View style={styles.codeArea}>
        {lines.map((line, i) => (
          <View key={i} style={styles.lineRow}>
            <Text style={styles.lineNum}>{i + 1}</Text>
            <Text style={styles.lineCode}>{highlightLine(line)}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.light.codeBackground,
    borderRadius: 12,
    overflow: "hidden",
    marginTop: 12,
  },
  langBadge: {
    alignSelf: "flex-end",
    paddingHorizontal: 10,
    paddingVertical: 4,
    backgroundColor: "rgba(255,255,255,0.08)",
    borderBottomLeftRadius: 8,
  },
  langText: {
    fontFamily: monoFont,
    fontSize: 10,
    color: Colors.light.codeComment,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  codeArea: {
    paddingHorizontal: 12,
    paddingVertical: 10,
    paddingTop: 4,
  },
  lineRow: {
    flexDirection: "row",
    minHeight: 18,
  },
  lineNum: {
    width: 24,
    fontFamily: monoFont,
    fontSize: 11,
    color: Colors.light.codeComment,
    textAlign: "right",
    marginRight: 12,
    lineHeight: 18,
  },
  lineCode: {
    fontFamily: monoFont,
    fontSize: 12,
    lineHeight: 18,
    flex: 1,
    color: Colors.light.codeText,
  },
});
