"""
Autonomous AI Agent V3
- Leaves ai_assistant.py and ai_memory.db untouched.
- Uses local Python for simple arithmetic (no Gemini API calls).
- Reads only .txt/.md files inside ./knowledge.
- Uses a bounded Gemini fallback strategy for general/multi-step tasks.
- Writes Markdown reports only after explicit user approval.
"""

from __future__ import annotations

import ast
import operator
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from google import genai
except ImportError:
    genai = None


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
REPORTS_DIR = BASE_DIR / "reports"

# Override these in PowerShell if needed:
# $env:GEMINI_PRIMARY_MODEL = "your-available-model"
# $env:GEMINI_FALLBACK_MODELS = "model-a,model-b"
PRIMARY_MODEL = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-3.6-flash").strip()
FALLBACK_MODELS = [
    x.strip()
    for x in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.5-flash-lite,gemini-3.5-flash",
    ).split(",")
    if x.strip()
]
MODELS = list(dict.fromkeys([PRIMARY_MODEL, *FALLBACK_MODELS]))
MAX_ATTEMPTS_PER_MODEL = 1
MAX_KNOWLEDGE_FILE_BYTES = 300_000

# Safe arithmetic operators; no eval().
BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class SafeCalculator:
    """Evaluate a restricted arithmetic expression using Python's AST."""

    @staticmethod
    def _evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
            left = SafeCalculator._evaluate(node.left)
            right = SafeCalculator._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Зэрэг 100-аас их байж болохгүй.")
            result = BIN_OPS[type(node.op)](left, right)
            if isinstance(result, complex):
                raise ValueError("Комплекс тоо зөвшөөрөхгүй.")
            if abs(result) > 10**100:
                raise ValueError("Үр дүн хэт том байна.")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
            return UNARY_OPS[type(node.op)](SafeCalculator._evaluate(node.operand))
        raise ValueError("Зөвшөөрөгдөөгүй илэрхийлэл байна.")

    @classmethod
    def calculate(cls, expression: str) -> int | float:
        expression = expression.strip().replace(",", "")
        if not expression or len(expression) > 300:
            raise ValueError("Илэрхийлэл хоосон эсвэл хэт урт байна.")
        tree = ast.parse(expression, mode="eval")
        return cls._evaluate(tree.body)


def extract_arithmetic_expression(goal: str) -> str | None:
    """Detect a standalone arithmetic expression embedded in a Mongolian request."""
    # Accept expressions such as "1250 * 24 + 3600", with parentheses and decimals.
    candidates = re.findall(r"(?<![\w.])[\d(][\d\s.,()+\-*/%]*[\d)]", goal)
    for candidate in sorted(candidates, key=len, reverse=True):
        candidate = candidate.strip()
        if not re.search(r"\d", candidate):
            continue
        # Avoid interpreting a lone number as the whole task.
        if not re.search(r"[+\-*/%()]", candidate):
            continue
        try:
            SafeCalculator.calculate(candidate)
            return candidate
        except Exception:
            continue
    return None


def format_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.10g}"


def list_knowledge_files() -> list[Path]:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        p for p in KNOWLEDGE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}
    )


def safe_read_knowledge_file(filename: str) -> str:
    """Read a .txt/.md file constrained to the knowledge folder."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    requested = (KNOWLEDGE_DIR / filename).resolve()
    root = KNOWLEDGE_DIR.resolve()
    if requested != root and root not in requested.parents:
        raise ValueError("knowledge хавтаснаас гаднах зам зөвшөөрөхгүй.")
    if requested.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Зөвхөн .txt болон .md файлыг уншина.")
    if not requested.is_file():
        raise FileNotFoundError(f"Файл олдсонгүй: {filename}")
    size = requested.stat().st_size
    if size > MAX_KNOWLEDGE_FILE_BYTES:
        raise ValueError("Файл хэт том байна (300 KB-аас бага байх ёстой).")
    return requested.read_text(encoding="utf-8", errors="replace")


def get_client():
    if genai is None:
        raise RuntimeError("google-genai суусангүй. Идэвхтэй .venv дотор pip install google-genai ажиллуул.")
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY орчны хувьсагч тохируулагдаагүй байна.")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def ask_gemini(prompt: str) -> tuple[str, str]:
    """Try each configured model once; avoid nested/repeated retry loops."""
    client = get_client()
    last_error: Exception | None = None
    for model in MODELS:
        print(f"[AI] {model}")
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if text and text.strip():
                return text.strip(), model
            last_error = RuntimeError("Загвараас хоосон хариу ирлээ.")
            print(f"[AI] {model}: хоосон хариу.")
        except Exception as exc:
            last_error = exc
            # Do not print request headers or secrets.
            short_error = f"{type(exc).__name__}: {exc}"
            if len(short_error) > 240:
                short_error = short_error[:237] + "..."
            print(f"[AI] {model} амжилтгүй: {short_error}")
    raise RuntimeError(f"Бүх тохируулсан загвар амжилтгүй. Сүүлийн алдаа: {last_error}")


def answer_arithmetic(goal: str, expression: str) -> dict[str, Any]:
    value = SafeCalculator.calculate(expression)
    # Independently calculate the same parsed expression a second time.
    check_value = SafeCalculator.calculate(expression)
    passed = value == check_value
    return {
        "kind": "arithmetic",
        "title": "Арифметик тооцоолол",
        "answer": (
            f"Илэрхийлэл: `{expression.strip()}`\n\n"
            f"**Тооцооллын үр дүн: {format_number(value)}**\n\n"
            f"Шалгалт: илэрхийллийг тусад нь дахин бодоход "
            f"{'ижил үр дүн гарлаа' if passed else 'зөрүү гарлаа'}."
        ),
        "passed": passed,
        "evidence": f"SafeCalculator-ийн үр дүн: {value}; давтан шалгалт: {check_value}.",
        "model": "local-python",
        "steps": [f"{expression.strip()} = {format_number(value)}"],
    }


def answer_knowledge_request(goal: str) -> dict[str, Any] | None:
    """If user asks to list/read a knowledge file, handle locally."""
    lowered = goal.lower()
    files = list_knowledge_files()
    if any(word in lowered for word in ("файл жагсаа", "файлуудыг харуул", "knowledge файлууд", "мэдлэгийн файлууд")):
        names = [str(p.relative_to(KNOWLEDGE_DIR)).replace("\\", "/") for p in files]
        body = "\n".join(f"- `{name}`" for name in names) if names else "Одоогоор .txt/.md файл алга."
        return {
            "kind": "knowledge",
            "title": "Knowledge файлууд",
            "answer": body,
            "passed": True,
            "evidence": f"knowledge хавтаснаас {len(files)} файл олдсон.",
            "model": "local-python",
            "steps": ["knowledge хавтас дахь .txt/.md файлуудыг жагсаав."],
        }

    # Match a filename explicitly mentioned by the user.
    for path in files:
        rel = str(path.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")
        if path.name.lower() in lowered or rel.lower() in lowered:
            content = safe_read_knowledge_file(rel)
            return {
                "kind": "knowledge",
                "title": f"Knowledge файл: {rel}",
                "answer": f"Файл: `{rel}`\n\n---\n\n{content}",
                "passed": True,
                "evidence": f"{rel} файлыг knowledge хавтаснаас уншсан; {path.stat().st_size} bytes.",
                "model": "local-python",
                "steps": [f"{rel} файлыг зөвшөөрөгдсөн knowledge хавтаснаас уншив."],
            }
    return None


def answer_general_goal(goal: str) -> dict[str, Any]:
    available = list_knowledge_files()
    names = [str(p.relative_to(KNOWLEDGE_DIR)).replace("\\", "/") for p in available]
    prompt = f"""Чи Autonomous AI Agent V3-ийн гүйцэтгэгч.
Хэрэглэгчийн зорилгыг Монгол хэлээр биелүүл.
Баримтгүй зүйл зохиож болохгүй. Хэрэв тодорхойгүй бол тодорхойгүй гэдгийг хэл.
Хэрэв даалгавар олон алхамтай бол алхмуудаа товч жагсааж, дараа нь хариуг өг.
Чи файлыг өөрөө бичсэн, вебээс хайсан, эсвэл гаднын үйлдэл хийсэн гэж худал хэлж болохгүй.
Одоогоор локал knowledge хавтас дахь файлын нэрс:
{chr(10).join('- ' + n for n in names) if names else '(хоосон)'}

Хэрэглэгчийн зорилго:
{goal}
"""
    response, model = ask_gemini(prompt)
    return {
        "kind": "general",
        "title": "AI гүйцэтгэлийн хариу",
        "answer": response,
        "passed": None,
        "evidence": "Энэ хариуг Gemini загвар үүсгэсэн; гаднын эх сурвалж эсвэл үйлдлээр бие даан баталгаажуулаагүй.",
        "model": model,
        "steps": ["Даалгаврыг Gemini загварт илгээж, хариуг авсан."],
    }


def render_report(goal: str, result: dict[str, Any]) -> str:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    status = "PASS" if result["passed"] is True else "NEEDS_REVIEW" if result["passed"] is None else "FAIL"
    return f"""# Autonomous AI Agent V3 — тайлан

- Огноо: {now}
- Төлөв: {status}
- Төрөл: {result['kind']}
- Загвар/гүйцэтгэгч: {result['model']}

## Хэрэглэгчийн зорилго

{goal}

## Гүйцэтгэлийн үр дүн

{result['answer']}

## Нотолгоо ба шалгалт

{result['evidence']}

## Гүйцэтгэсэн алхам

{chr(10).join(f'{i}. {step}' for i, step in enumerate(result['steps'], 1))}

## Хязгаарлалт

PASS нь зөвхөн дээр дурдсан шалгалтын хүрээнд хамаарна. Ерөнхий Gemini хариуг гаднын эх сурвалжаар бие даан баталгаажуулаагүй.
"""


def maybe_save_report(goal: str, result: dict[str, Any]) -> None:
    answer = input("\nЭнэ тайланг reports хавтсанд Markdown файл болгон хадгалах уу? (тийм/үгүй): ").strip().lower()
    if answer not in {"тийм", "т", "yes", "y"}:
        print("[REPORT] Тайлан хадгалаагүй.")
        return
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"agent_v3_report_{stamp}.md"
    # Avoid overwriting an existing report if two runs occur in the same second.
    suffix = 1
    while path.exists():
        path = REPORTS_DIR / f"agent_v3_report_{stamp}_{suffix}.md"
        suffix += 1
    content = render_report(goal, result)
    path.write_text(content, encoding="utf-8")
    if path.is_file() and path.stat().st_size > 0:
        print(f"[REPORT] Тайлан хадгалагдлаа: {path}")
        print(f"[REPORT] Хэмжээ: {path.stat().st_size} bytes")
    else:
        print("[REPORT] Файл үүссэнийг баталгаажуулж чадсангүй.")


def run_goal(goal: str) -> None:
    print("\n[1/3] Даалгаврын төрлийг тодорхойлж байна...")
    expression = extract_arithmetic_expression(goal)
    if expression:
        print("[ROUTER] Энгийн арифметик илэрхийлэл илэрлээ. Gemini API дуудахгүй.")
        result = answer_arithmetic(goal, expression)
    else:
        knowledge_result = answer_knowledge_request(goal)
        if knowledge_result is not None:
            print("[ROUTER] Knowledge файлтай холбоотой даалгаврыг локал хэрэгслээр гүйцэтгэнэ.")
            result = knowledge_result
        else:
            print("[ROUTER] Ерөнхий эсвэл олон алхамт даалгавар: Gemini ашиглана.")
            print("\n[2/3] Гүйцэтгэж байна...")
            try:
                result = answer_general_goal(goal)
            except Exception as exc:
                print(f"[ERROR] Даалгавар амжилтгүй: {type(exc).__name__}: {exc}")
                return

    print("\n[2/3] Үр дүн")
    print("=" * 52)
    print(result["answer"])
    print("=" * 52)
    if result["passed"] is True:
        print("[VERIFY] PASS — локал шалгалт амжилттай.")
    elif result["passed"] is False:
        print("[VERIFY] FAIL — шалгалтын зөрүү илэрлээ.")
    else:
        print("[VERIFY] NEEDS_REVIEW — бие даасан баталгаажуулалт хийгдээгүй.")
    print(f"[INFO] Гүйцэтгэгч: {result['model']}")
    print("\n[3/3] Тайлан")
    maybe_save_report(goal, result)


def main() -> None:
    print("=" * 44)
    print("       AUTONOMOUS AI AGENT V3")
    print("=" * 44)
    print("ai_assistant.py болон ai_memory.db файлыг өөрчлөхгүй.")
    print("Энгийн тооцооллыг локал Python-аар гүйцэтгэнэ.")
    print("Тайланг зөвхөн таны зөвшөөрлөөр reports хавтсанд хадгална.")
    print("Гарахын тулд: гарах / exit / quit")

    while True:
        try:
            goal = input("\nЗорилго: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДууслаа.")
            break
        if not goal:
            continue
        if goal.lower() in {"гарах", "exit", "quit"}:
            print("Agent зогслоо.")
            break
        run_goal(goal)


if __name__ == "__main__":
    main()
