# Palette's Journal - Critical Learnings

## 2026-10-30 - CLI Command UX Guidance
**Learning:** In interactive CLI applications, users frequently type `/help`, `help`, or `/?` to discover commands. Without explicit local handlers, CLI assistants pass these system requests to external APIs/LLMs, causing latency, costs, and non-standard responses.
**Action:** Always intercept system help commands (`/help`, `help`, `/?`) locally in CLI tools to provide instant, bilingual command documentation.
