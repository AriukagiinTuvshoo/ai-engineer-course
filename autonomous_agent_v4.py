"""
Autonomous AI Agent V4
- Bilingual Mongolian/English interface
- Real local text/Markdown file reading and grounded summaries
- Safe arithmetic calculator (AST allowlist; no eval)
- Optional planner/executor/verifier multi-agent workflow
- Uses a separate SQLite memory file; never opens or modifies ai_memory.db
- Does not edit ai_assistant.py or autonomous_agent_v3.py

Requirements:
    pip install google-genai

Environment variables:
    GEMINI_API_KEY              Required
    GEMINI_PRIMARY_MODEL        Default: gemini-3.5-flash-lite
    GEMINI_FALLBACK_MODELS      Default: gemini-3.5-flash
    AGENT_V4_MULTI_AGENT        "1" enables planner + executor + verifier (more API calls)
    AGENT_V4_MEMORY_DB          Default: agent_v4_memory.db (separate from ai_memory.db)
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from google import genai
except ImportError:
    genai = None


ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"
REPORTS_DIR = ROOT / "reports"
MEMORY_DB = ROOT / os.getenv("AGENT_V4_MEMORY_DB", "agent_v4_memory.db")
PRIMARY_MODEL = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-3.5-flash-lite").strip()
FALLBACK_MODELS = [
    x.strip()
    for x in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash").split(",")
    if x.strip()
]
MULTI_AGENT = os.getenv("AGENT_V4_MULTI_AGENT", "0").strip().lower() in {"1", "true", "yes"}
MAX_FILE_BYTES = 500_000
MAX_PROMPT_CHARS = 20_000


def detect_language(text: str) -> str:
    """Return 'mn' when Cyrillic letters are present, otherwise 'en'."""
    return "mn" if re.search(r"[А-Яа-яЁёӨөҮү]", text) else "en"


def safe_calculate(expression: str) -> float | int:
    """Evaluate arithmetic using an AST allowlist; never use eval()."""
    tree = ast.parse(expression, mode="eval")
    allowed_binary = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    allowed_unary = (ast.UAdd, ast.USub)

    def visit(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if abs(node.value) > 10**12:
                raise ValueError("Number too large.")
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, allowed_binary):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponent too large.")
            operations = {
                ast.Add: lambda: left + right,
                ast.Sub: lambda: left - right,
                ast.Mult: lambda: left * right,
                ast.Div: lambda: left / right,
                ast.FloorDiv: lambda: left // right,
                ast.Mod: lambda: left % right,
                ast.Pow: lambda: left ** right,
            }
            result = operations[type(node.op)]()
            if isinstance(result, (int, float)) and (not math.isfinite(result) or abs(result) > 10**15):
                raise ValueError("Result out of allowed range.")
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, allowed_unary):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError("Only basic arithmetic is allowed.")

    return visit(tree)


def extract_arithmetic(text: str) -> str | None:
    """Recognize explicit arithmetic requests without grabbing unrelated numbers."""
    patterns = [
        r"(?:тооцоол|бод|calculate|compute|what is)\s*[:：]?\s*([0-9][0-9\s().+\-*/%]+)",
        r"^\s*([0-9][0-9\s().+\-*/%]+)\s*(?:тооцоол|бод|calculate|compute)?\s*[?.!]*\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            expr = match.group(1).strip()
            expr = re.sub(r"\s+", "", expr)
            if re.search(r"[+\-*/%]", expr) and re.fullmatch(r"[0-9().+\-*/%]+", expr):
                return expr
    return None


def _safe_knowledge_path(filename: str) -> Path:
    name = Path(filename).name
    if name != filename or Path(filename).suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Only .txt/.md files directly inside the knowledge folder are allowed.")
    path = (KNOWLEDGE_DIR / name).resolve()
    if path.parent != KNOWLEDGE_DIR.resolve():
        raise ValueError("Path outside knowledge folder is not allowed.")
    return path


def list_knowledge_files() -> list[str]:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in KNOWLEDGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md"})


def read_knowledge_file(filename: str) -> str:
    path = _safe_knowledge_path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"File not found in knowledge/: {filename}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds {MAX_FILE_BYTES} byte safety limit.")
    return path.read_text(encoding="utf-8")[:MAX_PROMPT_CHARS]


def init_memory() -> None:
    """Create/use a dedicated V4 database; never connect to ai_memory.db."""
    MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(MEMORY_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                language TEXT NOT NULL,
                user_text TEXT NOT NULL,
                assistant_text TEXT NOT NULL
            )
        """)
        conn.commit()


def save_memory(user_text: str, assistant_text: str, language: str) -> None:
    init_memory()
    with sqlite3.connect(MEMORY_DB) as conn:
        conn.execute(
            "INSERT INTO conversations(created_at, language, user_text, assistant_text) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), language, user_text[:5000], assistant_text[:10000]),
        )
        conn.commit()


class GeminiClient:
    def __init__(self) -> None:
        if genai is None:
            raise RuntimeError("google-genai is missing. Run: pip install google-genai")
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY is not set in this PowerShell session.")
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.models = list(dict.fromkeys([PRIMARY_MODEL, *FALLBACK_MODELS]))

    def ask(self, prompt: str, system_instruction: str = "") -> tuple[str, str]:
        errors = []
        for model in self.models:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"system_instruction": system_instruction} if system_instruction else None,
                )
                answer = getattr(response, "text", None)
                if answer and answer.strip():
                    return answer.strip(), model
                errors.append(f"{model}: empty response")
            except Exception as exc:
                errors.append(f"{model}: {type(exc).__name__}: {exc}")
        raise RuntimeError("All Gemini models failed:\n" + "\n".join(errors))


def find_filename(text: str) -> str | None:
    match = re.search(r"(?<![\w.-])([A-Za-z0-9_. -]+\.(?:txt|md))\b", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def wants_summary(text: str) -> bool:
    return bool(re.search(r"\b(summarize|summary|explain|main idea|key points|describe)\b|гол\s*санаа|тайлбарла|дүгнэ|хураангуйл", text, re.IGNORECASE))


def build_system_instruction(language: str) -> str:
    lang = "Mongolian" if language == "mn" else "English"
    return (
        "You are a careful bilingual AI engineering assistant. "
        f"Answer in {lang} unless the user explicitly requests another language. "
        "Never claim to have read a file unless its contents are included in the prompt. "
        "Separate verified facts from assumptions. If uncertain, say so. "
        "Do not claim code was executed or tested unless an actual test result is supplied. "
        "For software plans, include concrete files, steps, tests, and acceptance criteria. "
        "Do not suggest editing ai_assistant.py, ai_memory.db, or autonomous_agent_v3.py."
    )


def run_mult_agent(client: GeminiClient, user_text: str, language: str, context: str = "") -> tuple[str, str]:
    """Optional three-role workflow. Each role is bounded to one call."""
    sys = build_system_instruction(language)
    plan, model1 = client.ask(
        "Create a short, actionable plan for this task. Do not claim to execute anything.\n"
        f"USER TASK:\n{user_text}\nCONTEXT:\n{context[:8000]}",
        sys + " You are the PLANNER role. Output concise steps and risks."
    )
    draft, model2 = client.ask(
        f"Carry out the user's request as far as text reasoning permits. Follow this plan:\n{plan}\n"
        f"USER TASK:\n{user_text}\nCONTEXT:\n{context[:8000]}",
        sys + " You are the EXECUTOR role. Do not claim filesystem changes or tests occurred."
    )
    review, model3 = client.ask(
        "Review this proposed answer. Identify unsupported claims, missing requirements, and test gaps. "
        "Then provide a corrected final answer. Do not pretend to run tests.\n"
        f"USER TASK:\n{user_text}\nPLAN:\n{plan}\nDRAFT:\n{draft}",
        sys + " You are the VERIFIER role. You review reasoning only; you cannot independently execute code."
    )
    return review, f"{model1}, {model2}, {model3}"


def answer_request(client: GeminiClient, user_text: str, language_override: str | None = None) -> tuple[str, str, str]:
    language = language_override or detect_language(user_text)
    expression = extract_arithmetic(user_text)
    if expression:
        try:
            result = safe_calculate(expression)
            answer = f"Тооцоолол: `{expression}` = **{result:,}**\nБаталгаажуулалт: локал Python арифметик тооцоолол амжилттай." if language == "mn" else f"Calculation: `{expression}` = **{result:,}**\nVerification: local Python arithmetic succeeded."
            return answer, "LOCAL_CALCULATOR", "PASS"

        except Exception as exc:
            msg = f"Тооцоолол боломжгүй: {exc}" if language == "mn" else f"Calculation failed: {exc}"
            return msg, "LOCAL_CALCULATOR", "FAIL"

    context = ""
    filename = find_filename(user_text)
    if filename:
        try:
            content = read_knowledge_file(filename)
            context = f"ACTUAL FILE CONTENT: {filename}\n---\n{content}\n---"
            if wants_summary(user_text):
                prompt = (
                    f"The user asked about the real local file {filename}. Use ONLY the supplied file contents for file-specific claims. "
                    "Summarize the key ideas, explain important terms, and clearly say if the file lacks information.\n\n"
                    f"USER REQUEST:\n{user_text}\n\n{context}"
                )
            else:
                prompt = f"Answer the user based on this actual file content. Quote only short relevant excerpts.\nUSER: {user_text}\n{context}"
        except (FileNotFoundError, ValueError, OSError) as exc:
            msg = f"Файл уншиж чадсангүй: {exc}" if language == "mn" else f"Could not read file: {exc}"
            return msg, "LOCAL_FILE_READER", "FAIL"
    else:
        prompt = (
            f"User request:\n{user_text}\n\n"
            f"Available local knowledge files: {', '.join(list_knowledge_files()) or '(none)'}\n"
            "Do not claim to have opened files unless their contents are provided."
        )

    try:
        if MULTI_AGENT:
            answer, model = run_mult_agent(client, user_text, language, context)
            status = "NEEDS_REVIEW"  # verifier is another model call, not independent execution
        else:
            answer, model = client.ask(prompt, build_system_instruction(language))
            status = "NEEDS_REVIEW"  # model answer not independently verified
        return answer, model, status
    except Exception as exc:
        msg = f"Gemini API алдаа: {exc}" if language == "mn" else f"Gemini API error: {exc}"
        return msg, "GEMINI_API", "FAIL"


def write_report(user_text: str, answer: str, model: str, status: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"agent_v4_report_{stamp}.md"
    suffix = 1
    while path.exists():
        path = REPORTS_DIR / f"agent_v4_report_{stamp}_{suffix}.md"
        suffix += 1
    content = (
        "# Autonomous AI Agent V4 Report\n\n"
        f"- Created: {datetime.now().isoformat(timespec='seconds')}\n"
        f"- Model/Tool: {model}\n- Status: {status}\n\n"
        "## User request\n\n" + user_text + "\n\n## Result\n\n" + answer + "\n"
    )
    path.write_text(content, encoding="utf-8")
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError("Report file verification failed.")
    return path


def main() -> None:
    print("=" * 44)
    print("       AUTONOMOUS AI AGENT V4")
    print("=" * 44)
    print("MN/EN bilingual • local file reader • safe calculator")
    print("Existing ai_assistant.py, ai_memory.db, autonomous_agent_v3.py are not modified.")
    print("Commands: /help | /lang mn | /lang en | /lang auto | /files | гарах / exit / quit")
    print(f"Primary model: {PRIMARY_MODEL}; multi-agent mode: {'ON' if MULTI_AGENT else 'OFF'}")
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    init_memory()
    language_override = None
    try:
        client = GeminiClient()
    except Exception as exc:
        print(f"[STARTUP ERROR] {exc}")
        return

    while True:
        try:
            user_text = input("\nGoal / Зорилго: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye / Баяртай.")
            break
        if not user_text:
            continue
        if user_text.lower() in {"гарах", "exit", "quit"}:
            print("Bye / Баяртай.")
            break
        if user_text in {"/help", "/?"}:
            lang = language_override or detect_language(user_text)
            if lang == "mn":
                print("\n=== Тусламж ба Ашиглах заавар 🎨 ===")
                print("Командууд:")
                print("  /help           - Энэхүү тусламжийн цэсийг харуулах")
                print("  /files          - 'knowledge/' хавтас дахь файлуудыг жагсаах")
                print("  /lang mn        - Хариултыг Монгол хэлээр авах")
                print("  /lang en        - Хариултыг Англи хэлээр авах")
                print("  /lang auto      - Хэлийг автоматаар тохируулах")
                print("  гарах / exit    - Программаас гарах")
                print("\nЖишээ асуултууд:")
                print("  - Тооцоолол: '1250 * 24 + 3600'")
                print("  - Файл унших: 'ai_engineering_notes.txt файлыг уншаад тайлбарла'")
            else:
                print("\n=== Help & Usage Guide 🎨 ===")
                print("Commands:")
                print("  /help           - Display this help menu")
                print("  /files          - List files in the 'knowledge/' folder")
                print("  /lang mn        - Force response language to Mongolian")
                print("  /lang en        - Force response language to English")
                print("  /lang auto      - Auto-detect response language")
                print("  exit / quit     - Exit the application")
                print("\nExample queries:")
                print("  - Calculation: 'Calculate 1250 * 24 + 3600'")
                print("  - File reading: 'Summarize ai_engineering_notes.txt'")
            continue
        if user_text == "/lang mn":
            language_override = "mn"
            print("Хариултын хэл: Монгол")
            continue
        if user_text == "/lang en":
            language_override = "en"
            print("Response language: English")
            continue
        if user_text == "/lang auto":
            language_override = None
            print("Language: automatic / Хэл: автоматаар")
            continue
        if user_text == "/files":
            files = list_knowledge_files()
            if files:
                print(f"\n📂 Knowledge Folder Files ({len(files)} file{'s' if len(files) > 1 else ''}):")
                for f in files:
                    print(f"  • {f}")
                print("\n💡 Tip: Mention any of these filenames in your prompt to read/summarize its contents.")
            else:
                print("\n📂 Knowledge folder is empty or has no .txt/.md files.")
                print("💡 Tip: Add .txt or .md files to 'knowledge/' directory to query them.")
            continue

        print("\n[RUN] Processing request...")
        answer, model, status = answer_request(client, user_text, language_override)
        print("\n" + "=" * 44)
        print(answer)
        print("=" * 44)
        print(f"[VERIFY] {status} — " + (
            "local arithmetic was computed." if status == "PASS"
            else "this is not an independent verification of the AI answer." if status == "NEEDS_REVIEW"
            else "request did not complete successfully."
        ))
        print(f"[INFO] Executor: {model}")

        try:
            consent = input("\nSave this result to reports/ as Markdown? (yes/no; тийм/үгүй): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            consent = "no"
        if consent in {"yes", "y", "тийм", "т", "да"}:
            try:
                path = write_report(user_text, answer, model, status)
                print(f"[REPORT] Saved and checked: {path.relative_to(ROOT)}")
            except OSError as exc:
                print(f"[REPORT ERROR] {exc}")
        else:
            print("[REPORT] Not saved.")

        # Save conversation only to the dedicated V4 DB, with explicit opt-in.
        try:
            remember = input("Save this conversation to the separate V4 memory? (yes/no; тийм/үгүй): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            remember = "no"
        if remember in {"yes", "y", "тийм", "т", "да"}:
            save_memory(user_text, answer, language_override or detect_language(user_text))
            print(f"[MEMORY] Saved to separate database: {MEMORY_DB.name}")


if __name__ == "__main__":
    main()
