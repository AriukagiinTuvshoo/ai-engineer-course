# Autonomous AI Agent V4

## Safety boundaries
- This version is a new standalone file.
- It does not import or modify `ai_assistant.py`, `ai_memory.db`, or `autonomous_agent_v3.py`.
- Conversation memory is OFF unless you explicitly opt in after each turn, and uses `agent_v4_memory.db`.
- Report writing also requires explicit consent.
- Multi-agent mode is optional and costs additional API calls. Its verifier is an LLM review, not independent execution.

## Install / run (PowerShell)
```powershell
# Run inside C:\Users\enhtv\Documents\ai-engineer-course with your existing .venv activated
python -m pip install google-genai
python -m unittest -v test_autonomous_agent_v4.py
python .\autonomous_agent_v4.py
```

Set the model (optional):
```powershell
$env:GEMINI_PRIMARY_MODEL = "gemini-3.5-flash-lite"
$env:GEMINI_FALLBACK_MODELS = "gemini-3.5-flash"
```

Enable planner/executor/reviewer roles (more API calls):
```powershell
$env:AGENT_V4_MULTI_AGENT = "1"
python .\autonomous_agent_v4.py
```

Language commands:
- `/lang mn` — answer in Mongolian
- `/lang en` — answer in English
- `/lang auto` — infer language from the user's message
- `/files` — list `.txt` and `.md` files under `knowledge/`
- `/help` — show bilingual help menu

## First acceptance checks
1. `Calculate 1250 * 24 + 3600` returns 33,600 and `PASS` from local calculation.
2. `ai_engineering_notes.txt файлыг уншаад гол санааг тайлбарла` passes the actual file contents to Gemini.
3. `Read and summarize ai_engineering_notes.txt in English` returns English.
4. `Make a 7 day plan with 60 minutes each day` must not be misclassified as arithmetic.
5. A path such as `../ai_memory.db` is rejected.
6. The existing `ai_assistant.py`, `ai_memory.db`, and `autonomous_agent_v3.py` remain untouched.
7. AI-generated responses remain `NEEDS_REVIEW`; only local arithmetic gets `PASS` in this version.
