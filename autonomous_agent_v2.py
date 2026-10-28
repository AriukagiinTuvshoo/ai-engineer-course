from __future__ import annotations

import ast
import json
import operator
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors, types


# ============================================================
# AUTONOMOUS AI AGENT V2
# Safe update: this file does not modify ai_assistant.py or
# write to ai_memory.db. Reports are saved only after approval.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "ai_memory.db"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
REPORTS_DIR = BASE_DIR / "reports"

PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_MODELS = [
    item.strip()
    for item in os.environ.get(
        "GEMINI_FALLBACK",
        "gemini-3.5-flash-lite,gemini-3.5-flash",
    ).split(",")
    if item.strip()
]

MAX_STEPS = 3
MAX_TOOL_ROUNDS = 2
RETRIES_PER_MODEL = 1
MAX_FILE_BYTES = 100_000
MAX_RESULT_CHARS = 12_000

SYSTEM_PROMPT = """
Чи бол хэрэглэгчийн зорилгыг биелүүлэхэд туслах болгоомжтой AI агент.
Зөвхөн зарласан хэрэгслүүдийг ашигла. Хийгээгүй үйлдлээ хийсэн гэж бүү хэл.
Файл, өгөгдлийн санг өөрчлөх хэрэгсэл байхгүй бол өөрчилсөн гэж бүү мэдэгд.
Хэрэгслийн үр дүнг нотолгоо гэж үз; таамаглалыг баримт мэт бүү танилцуул.
Хариултаа Монгол хэлээр, тодорхой бөгөөд товч өг.
"""


# -------------------------
# Retry and model fallback
# -------------------------

def is_retryable(error: Exception) -> bool:
    if isinstance(error, errors.ServerError):
        return True
    if isinstance(error, errors.ClientError):
        return getattr(error, "code", None) == 429
    return False


def generate_with_retry(client: genai.Client, **kwargs):
    requested_model = kwargs.pop("model", PRIMARY_MODEL)
    models = list(dict.fromkeys([requested_model] + FALLBACK_MODELS))
    last_error: Exception | None = None

    for model in models:
        for attempt in range(RETRIES_PER_MODEL + 1):
            try:
                print(f"[AI] {model} ({attempt + 1}/{RETRIES_PER_MODEL + 1})")
                return client.models.generate_content(model=model, **kwargs)
            except Exception as exc:
                last_error = exc
                if not is_retryable(exc):
                    raise
                if attempt < RETRIES_PER_MODEL:
                    delay = 1 + attempt
                    print(f"[AI] Түр зуурын алдаа. {delay} секунд хүлээнэ.")
                    time.sleep(delay)
                else:
                    print(f"[AI] {model} амжилтгүй; дараагийн загварыг туршина.")

    raise RuntimeError(f"Бүх AI загварын хүсэлт амжилтгүй боллоо: {last_error}")


# -------------------------
# Read-only local tools
# -------------------------

def read_saved_memories() -> str:
    """Read memories without creating or modifying the database."""
    if not DB_PATH.is_file():
        return "ai_memory.db файл олдсонгүй."

    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
            ).fetchone()
            if not table:
                return "memories хүснэгт олдсонгүй."
            rows = conn.execute(
                "SELECT key, value FROM memories ORDER BY key"
            ).fetchall()
    except sqlite3.Error as exc:
        return f"Санах ойг уншихад алдаа гарлаа: {exc}"

    if not rows:
        return "Хадгалсан санах ой хоосон байна."
    return "\n".join(f"{key}: {value}" for key, value in rows)


def list_knowledge_files() -> str:
    if not KNOWLEDGE_DIR.is_dir():
        return "knowledge хавтас олдсонгүй."
    names = sorted(
        p.name for p in KNOWLEDGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}
    )
    return "\n".join(names) if names else "knowledge хавтсанд .txt эсвэл .md файл алга."


def read_knowledge_file(filename: str) -> str:
    if not isinstance(filename, str) or not filename.strip():
        return "Алдаа: файлын нэр хоосон байна."

    filename = filename.strip()
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        return "Алдаа: зөвхөн knowledge хавтас доторх файлын нэрийг ашиглаж болно."
    if filename in {".", ".."}:
        return "Алдаа: файлын нэр буруу байна."

    root = KNOWLEDGE_DIR.resolve()
    path = (root / filename).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return "Алдаа: зөвшөөрөгдөөгүй зам."

    if path.suffix.lower() not in {".txt", ".md"}:
        return "Зөвхөн .txt болон .md файлыг уншина."
    if not path.is_file():
        return f"Файл олдсонгүй: {filename}"

    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return "Файлын хэмжээ 100 KB-аас их байна."
        return path.read_text(encoding="utf-8-sig")[:MAX_RESULT_CHARS]
    except (OSError, UnicodeError) as exc:
        return f"Файл уншихад алдаа гарлаа: {exc}"


# -------------------------
# Safe arithmetic
# -------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _calculate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("Зөвхөн тоо ашиглана.")
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Энэ үйлдлийг зөвшөөрөхгүй.")
        left = _calculate_node(node.left)
        right = _calculate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Зэрэгт дэвшүүлэх хэмжээ 8-аас их байж болохгүй.")
        result = _ALLOWED_OPERATORS[op_type](left, right)
        if not isinstance(result, (int, float)) or abs(result) > 1e100:
            raise ValueError("Үр дүн хэт том байна.")
        return result

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Энэ үйлдлийг зөвшөөрөхгүй.")
        return _ALLOWED_OPERATORS[op_type](_calculate_node(node.operand))

    raise ValueError("Зөвшөөрөгдөөгүй математик илэрхийлэл.")


def calculate(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 200:
        return "Алдаа: математик илэрхийлэл хоосон эсвэл хэт урт байна."
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_calculate_node(tree.body))
    except Exception as exc:
        return f"Тооцооллын алдаа: {exc}"


# -------------------------
# Function declarations
# -------------------------

TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="read_saved_memories",
            description="Хэрэглэгчийн SQLite санах ойг зөвхөн унших.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="list_knowledge_files",
            description="knowledge хавтас дахь .txt болон .md файлуудын нэрийг жагсаах.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="read_knowledge_file",
            description="knowledge хавтас доторх нэг .txt эсвэл .md файлыг унших.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "filename": types.Schema(
                        type=types.Type.STRING,
                        description="Файлын нэр, жишээ нь ai_engineering_notes.txt",
                    )
                },
                required=["filename"],
            ),
        ),
        types.FunctionDeclaration(
            name="calculate",
            description="Аюулгүй математик илэрхийлэл бодох. Жишээ: (125 + 75) * 3",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(type=types.Type.STRING)
                },
                required=["expression"],
            ),
        ),
    ]
)


def execute_tool(name: str, args: dict[str, Any]) -> str:
    if name == "read_saved_memories":
        return read_saved_memories()
    if name == "list_knowledge_files":
        return list_knowledge_files()
    if name == "read_knowledge_file":
        return read_knowledge_file(args.get("filename", ""))
    if name == "calculate":
        return calculate(args.get("expression", ""))
    return f"Зөвшөөрөгдөөгүй хэрэгсэл: {name}"


# -------------------------
# Planner and executor
# -------------------------

def create_plan(client: genai.Client, goal: str) -> list[str]:
    prompt = f"""
{SYSTEM_PROMPT}

Хэрэглэгчийн зорилго:
{goal}

Зорилгод хүрэх хамгийн цөөн, бодитой алхам гарга. 1-3 алхмаас хэтрүүлж болохгүй.
Энгийн тооцоог тус тусын олон алхам болгон хуваахгүй.
Файлд тайлан хадгалахыг хүссэн бол тайланг зөвхөн хэрэглэгч тусад нь зөвшөөрсний дараа хадгална.
Хэрэгслийн боломжгүй үйлдлийг хийж чадна гэж төлөвлөхгүй.
Зөвхөн дараах бүтэцтэй JSON буцаа:
{{"steps": ["алхам 1", "алхам 2"]}}
"""
    response = generate_with_retry(
        client,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    try:
        data = json.loads(response.text or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("AI төлөвлөгөөний JSON уншигдсангүй.") from exc

    raw_steps = data.get("steps", [])
    if not isinstance(raw_steps, list):
        raise ValueError("Төлөвлөгөөний steps нь жагсаалт байх ёстой.")
    steps = [s.strip() for s in raw_steps if isinstance(s, str) and s.strip()]
    if not steps:
        raise ValueError("AI хэрэгжүүлэх алхам гаргасангүй.")
    return steps[:MAX_STEPS]


def execute_step(
    client: genai.Client,
    goal: str,
    step: str,
    previous_results: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> str:
    prompt = f"""
{SYSTEM_PROMPT}

Хэрэглэгчийн зорилго:
{goal}

Одоо хийх алхам:
{step}

Өмнөх алхмуудын үр дүн:
{json.dumps(previous_results, ensure_ascii=False)}

Заавар:
- Шаардлагатай бол зарласан хэрэгслүүдийг ашигла.
- Хэрэгслийн үр дүнг өөрчлөлгүй нотолгоо болгон авч үз.
- Хэрэгсэл дуудаагүй бол бодит файл/өгөгдлийн сан өөрчилсөн гэж бүү мэдэгд.
- Тайланг файлд хадгалах шаардлагатай бол одоогоор зөвхөн тайлангийн агуулгыг бэлтгэ.
- Эцсийн хариултаа Монгол хэлээр өг.
"""
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]
    config = types.GenerateContentConfig(
        tools=[TOOL],
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )

    for _ in range(MAX_TOOL_ROUNDS + 1):
        response = generate_with_retry(client, contents=contents, config=config)
        if not response.candidates or not response.candidates[0].content:
            raise RuntimeError("AI-аас хоосон хариу ирлээ.")

        model_content = response.candidates[0].content
        contents.append(model_content)
        calls = response.function_calls or []

        if not calls:
            if not response.text:
                raise RuntimeError("AI текстэн хариу буцаасангүй.")
            return response.text

        parts = []
        for call in calls:
            args = dict(call.args or {})
            print(f"  [TOOL] {call.name} {args if args else ''}")
            tool_result = execute_tool(call.name, args)
            evidence.append({
                "tool": call.name,
                "arguments": args,
                "result": tool_result,
            })
            parts.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": tool_result},
                )
            )
        contents.append(types.Content(role="user", parts=parts))

    raise RuntimeError("Хэрэгслийн дуудлагын хязгаарт хүрлээ.")


# -------------------------
# Verification
# -------------------------

def verify_step(
    client: genai.Client,
    goal: str,
    step: str,
    result: str,
    evidence: list[dict[str, Any]],
) -> tuple[str, str]:
    # Tool output is concrete evidence. The model must not claim an action
    # happened unless the evidence supports it.
    relevant_evidence = evidence[-8:]
    prompt = f"""
{SYSTEM_PROMPT}

Алхам:
{step}

AI-ийн тайлбар:
{result}

Бодит хэрэгслийн нотолгоо (байхгүй бол хоосон жагсаалт):
{json.dumps(relevant_evidence, ensure_ascii=False)}

Шалгуур:
- Зөвхөн AI-ийн өөрийн мэдэгдлийг бодит гүйцэтгэлийн нотолгоо гэж бүү үз.
- Файл хадгалсан гэж хэлсэн боловч файл бичсэн нотолгоо байхгүй бол RETRY.
- Тооцооллын хэрэгслийн үр дүн байгаа бол тэр үр дүнг баримт гэж үз.
- Алхам нь зөвхөн тайлбар/төлөвлөлт бол хариулт зорилгод нийцэж байгаа эсэхийг үнэл.
JSON буцаа:
{{"status":"PASS" эсвэл "RETRY","reason":"Монгол хэлээр товч тайлбар"}}
"""
    response = generate_with_retry(
        client,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    try:
        data = json.loads(response.text or "{}")
    except json.JSONDecodeError:
        return "RETRY", "Шалгагчийн JSON буруу тул баталгаажуулж чадсангүй."

    status = data.get("status", "RETRY")
    if status not in {"PASS", "RETRY"}:
        status = "RETRY"
    reason = str(data.get("reason", "Шалтгаан ирсэнгүй."))

    # Guard against unsubstantiated claims about file creation.
    if re.search(r"(хадгалсан|үүсгэсэн|бичсэн).{0,30}(файл|тайлан)", result, re.IGNORECASE):
        if not any(item["tool"] == "write_report" for item in relevant_evidence):
            return "RETRY", "Файл бичсэн бодит хэрэгсэл ажиллаагүй; зөвхөн текстээр мэдэгдсэн байна."

    return status, reason


# -------------------------
# Report generation and save
# -------------------------

def build_report(goal: str, results: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> str:
    lines = [
        "# Autonomous AI Agent тайлан",
        "",
        f"- Огноо: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Зорилго: {goal}",
        "",
        "## Алхам ба үр дүн",
    ]
    for i, item in enumerate(results, start=1):
        lines.extend([
            "",
            f"### {i}. {item['step']}",
            f"- Төлөв: {item['status']}",
            f"- Баталгаажуулалт: {item['verification']}",
            "",
            item["result"],
        ])
    lines.extend(["", "## Хэрэгслийн нотолгоо"])
    if evidence:
        for item in evidence:
            lines.extend([
                "",
                f"- Хэрэгсэл: `{item['tool']}`",
                f"- Аргумент: `{json.dumps(item['arguments'], ensure_ascii=False)}`",
                f"- Үр дүн: {item['result']}",
            ])
    else:
        lines.append("\nЭнэ ажилд хэрэгсэл дуудаагүй.")
    return "\n".join(lines).strip() + "\n"


def save_report_after_approval(report: str) -> Path | None:
    answer = input("\nЭнэ тайланг reports хавтсанд Markdown файл болгон хадгалах уу? (тийм/үгүй): ")
    if answer.strip().lower() not in {"тийм", "tiim", "yes", "y"}:
        print("[REPORT] Тайлан файлд хадгалаагүй.")
        return None

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path = (REPORTS_DIR / filename).resolve()
    try:
        path.relative_to(REPORTS_DIR.resolve())
    except ValueError as exc:
        raise RuntimeError("Тайлангийн зам зөвшөөрөгдөөгүй.") from exc

    path.write_text(report, encoding="utf-8")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Тайлангийн файл үүссэн эсэхийг баталгаажуулж чадсангүй.")
    print(f"[REPORT] Тайлан хадгалагдлаа: {path}")
    print(f"[REPORT] Хэмжээ: {path.stat().st_size} bytes")
    return path


# -------------------------
# Main agent loop
# -------------------------

def run_agent(client: genai.Client, goal: str) -> None:
    print("\n[1/4] Төлөвлөгөө гаргаж байна...")
    steps = create_plan(client, goal)

    print("\n--- ТӨЛӨВЛӨГӨӨ ---")
    for i, step in enumerate(steps, start=1):
        print(f"{i}. {step}")

    approval = input("\nЭнэ төлөвлөгөөг хэрэгжүүлэх үү? (тийм/үгүй): ").strip().lower()
    if approval not in {"тийм", "tiim", "yes", "y"}:
        print("Төлөвлөгөөг цуцаллаа.")
        return

    results: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []

    for index, step in enumerate(steps, start=1):
        print(f"\n[2/4] Алхам {index}/{len(steps)}")
        print(f"Зорилт: {step}")
        try:
            result = execute_step(client, goal, step, results, evidence)
            print("\n[3/4] Үр дүнг шалгаж байна...")
            status, reason = verify_step(client, goal, step, result, evidence)

            # One retry only when verifier asks for it.
            if status == "RETRY":
                print(f"[VERIFY] Дахин оролдох шалтгаан: {reason}")
                retry_result = execute_step(client, goal, step, results, evidence)
                retry_status, retry_reason = verify_step(
                    client, goal, step, retry_result, evidence
                )
                result = retry_result
                status, reason = retry_status, retry_reason

            results.append({
                "step": step,
                "status": status,
                "result": result,
                "verification": reason,
            })
            print(f"[VERIFY] {status}: {reason}")
        except Exception as exc:
            print(f"[ERROR] Алхам амжилтгүй: {exc}")
            results.append({
                "step": step,
                "status": "ERROR",
                "result": str(exc),
                "verification": "Алхамд алдаа гарсан; амжилттай гэж батлаагүй.",
            })

    print("\n[4/4] Эцсийн тайлан")
    print("=" * 60)
    for i, item in enumerate(results, start=1):
        print(f"\n{i}. {item['step']}")
        print(f"Төлөв: {item['status']}")
        print(f"Шалгалт: {item['verification']}")
        print(f"Үр дүн:\n{item['result']}")

    passed = sum(item["status"] == "PASS" for item in results)
    print("\n" + "=" * 60)
    print(f"PASS алхам: {passed}/{len(results)}")
    if passed == len(results):
        print("Бүх алхмыг шалгагч PASS гэж үнэллээ. Энэ нь бүх төрлийн үйлдлийн бүрэн баталгаа биш.")
    else:
        print("Зарим алхам баталгаажаагүй. Үр дүнг дахин шалгаарай.")

    report = build_report(goal, results, evidence)
    try:
        save_report_after_approval(report)
    except Exception as exc:
        print(f"[REPORT ERROR] Тайлан хадгалахад алдаа гарлаа: {exc}")
        print("Тайлангийн агуулга дэлгэц дээр байна; файл хадгалалт баталгаажаагүй.")


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY орчны хувьсагч тохируулагдаагүй байна.")
        return

    client = genai.Client(api_key=api_key)
    print("\n====================================")
    print("       AUTONOMOUS AI AGENT V2")
    print("====================================")
    print("Энэ хувилбар ai_assistant.py болон ai_memory.db файлыг өөрчлөхгүй.")
    print("Тайланг зөвхөн таны зөвшөөрлөөр reports хавтсанд хадгална.")
    print("Гарахын тулд: гарах / exit / quit")

    while True:
        try:
            goal = input("\nЗорилго: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nПрограмыг хаалаа.")
            break

        if not goal:
            continue
        if goal.lower() in {"гарах", "exit", "quit"}:
            print("Програмыг хаалаа.")
            break

        try:
            run_agent(client, goal)
        except Exception as exc:
            print(f"\nАгентын алдаа: {exc}")


if __name__ == "__main__":
    main()
