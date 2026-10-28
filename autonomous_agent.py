
import ast
import json
import operator
import os
import time
from pathlib import Path

from google import genai
from google.genai import errors, types


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "ai_memory.db"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

PRIMARY_MODEL = os.environ.get(
    "GEMINI_MODEL", "gemini-3.6-flash"
)

FALLBACK_MODELS = [
    item.strip()
    for item in os.environ.get(
        "GEMINI_FALLBACK",
        "gemini-3.5-flash,gemini-3.5-flash-lite",
    ).split(",")
    if item.strip()
]

MAX_STEPS = 5
MAX_TOOL_ROUNDS = 3
MAX_RETRIES = 2
MAX_FILE_BYTES = 100_000

SYSTEM_PROMPT = """
Чи бол AI инженерийн ажлыг гүйцэтгэхэд тусалдаг AI агент.

Өгсөн зорилгыг ойлгож, жижиг алхмууд болгон төлөвлө.
Алхам бүрт шаардлагатай бол зөвшөөрөгдсөн хэрэгслүүдийг ашигла.
Бодит нотолгоонд үндэслэж дүгнэлт гарга.
Файл уншаагүй бол уншсан гэж, шалгаагүй бол шалгасан гэж бүү хэл.
Хэрэглэгчийн санах ой болон мэдлэгийн файлын мэдээллийг
хооронд нь ялгаж ойлго.
Хэрэглэгчтэй монгол хэлээр харилц.
Хэрэглэгчийн зөвшөөрөлгүйгээр файл өөрчлөхгүй.
"""


# ==========================================
# 2. RETRY AND FALLBACK
# ==========================================

def is_retryable(error):
    if isinstance(error, errors.ServerError):
        return True

    if isinstance(error, errors.ClientError):
        return getattr(error, "code", None) == 429

    return False


def generate_with_retry(client, **kwargs):
    last_error = None
    models = list(dict.fromkeys(
        [kwargs.pop("model", PRIMARY_MODEL)] + FALLBACK_MODELS
    ))

    for model in models:
        for attempt in range(MAX_RETRIES + 1):
            try:
                print(
                    f"[AI] {model} "
                    f"({attempt + 1}/{MAX_RETRIES + 1})"
                )

                return client.models.generate_content(
                    model=model,
                    **kwargs,
                )

            except Exception as exc:
                last_error = exc

                if not is_retryable(exc):
                    raise

                if attempt < MAX_RETRIES:
                    delay = 2 ** attempt
                    print(f"[AI] Түр зуурын алдаа. {delay} сек хүлээнэ.")
                    time.sleep(delay)

                else:
                    print(f"[AI] {model} загвар амжилтгүй.")

    raise RuntimeError(
        f"Бүх загварын хүсэлт амжилтгүй: {last_error}"
    )


# ==========================================
# 3. SAFE TOOLS
# ==========================================

def read_saved_memories():
    """Одоогийн SQLite санах ойг зөвхөн уншина."""

    import sqlite3

    if not DB_PATH.is_file():
        return "Санах ойн өгөгдлийн сан олдсонгүй."

    with sqlite3.connect(DB_PATH) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='memories'"
        ).fetchone()

        if not tables:
            return "Санах ойн хүснэгт олдсонгүй."

        rows = conn.execute(
            "SELECT key, value FROM memories ORDER BY key"
        ).fetchall()

    if not rows:
        return "Санах ой хоосон байна."

    return "\n".join(f"{key}: {value}" for key, value in rows)


def list_knowledge_files():
    """Мэдлэгийн хавтас дахь зөвшөөрөгдсөн файлуудыг жагсаана."""

    if not KNOWLEDGE_DIR.is_dir():
        return "knowledge хавтас олдсонгүй."

    files = []

    for path in KNOWLEDGE_DIR.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() in {".txt", ".md"}
        ):
            files.append(path.name)

    if not files:
        return "Унших боломжтой .txt эсвэл .md файл алга."

    return "\n".join(sorted(files))


def read_knowledge_file(filename):
    """knowledge хавтас дахь нэг текст файлыг уншина."""

    if not isinstance(filename, str) or not filename.strip():
        return "Алдаа: файлын нэр шаардлагатай."

    filename = filename.strip()

    if (
        Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or filename in {".", ".."}
    ):
        return "Алдаа: зөвхөн файлын нэр өгнө үү."

    root = KNOWLEDGE_DIR.resolve()
    path = (root / filename).resolve()

    try:
        path.relative_to(root)
    except ValueError:
        return "Алдаа: зөвшөөрөөгүй файлын зам."

    if path.suffix.lower() not in {".txt", ".md"}:
        return "Зөвхөн .txt болон .md файл уншина."

    if not path.is_file():
        return f"Файл олдсонгүй: {filename}"

    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return "Файл 100 KB хязгаараас том байна."

        return path.read_text(encoding="utf-8-sig")

    except (OSError, UnicodeError) as exc:
        return f"Файл унших алдаа: {exc}"


# Зөвхөн тоон арифметикийн хязгаарлагдмал үйлдэл.
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


def _calculate_node(node):
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("Зөвхөн тоо ашиглана.")
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)

        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Зөвшөөрөөгүй математик үйлдэл.")

        left = _calculate_node(node.left)
        right = _calculate_node(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Зэрэг 8-аас их байж болохгүй.")

        result = _ALLOWED_OPERATORS[op_type](left, right)

        if abs(result) > 1e100:
            raise ValueError("Тоо хэт их байна.")

        return result

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)

        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Зөвшөөрөөгүй үйлдэл.")

        return _ALLOWED_OPERATORS[op_type](
            _calculate_node(node.operand)
        )

    raise ValueError("Зөвхөн энгийн арифметик илэрхийлэл зөвшөөрнө.")


def calculate(expression):
    """Python код ажиллуулахгүйгээр арифметик тооцоолно."""

    if not isinstance(expression, str) or len(expression) > 200:
        return "Алдаа: илэрхийлэл хоосон эсвэл хэт урт байна."

    try:
        tree = ast.parse(expression, mode="eval")
        result = _calculate_node(tree.body)
        return str(result)
    except Exception as exc:
        return f"Тооцооллын алдаа: {exc}"


# ==========================================
# 4. FUNCTION CALLING
# ==========================================

TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="read_saved_memories",
            description="Хэрэглэгчийн одоо байгаа SQLite санах ойг уншина.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="list_knowledge_files",
            description="knowledge хавтас дахь текст файлуудыг жагсаана.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="read_knowledge_file",
            description="knowledge хавтас дахь нэг .txt эсвэл .md файлыг уншина.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "filename": types.Schema(
                        type=types.Type.STRING,
                        description="Файлын нэр, жишээ: ai_engineering_notes.txt",
                    )
                },
                required=["filename"],
            ),
        ),
        types.FunctionDeclaration(
            name="calculate",
            description="Энгийн математик илэрхийлэл бодно.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(
                        type=types.Type.STRING,
                        description="Жишээ: (125 + 75) * 3",
                    )
                },
                required=["expression"],
            ),
        ),
    ]
)


def execute_tool(name, args):
    if name == "read_saved_memories":
        return read_saved_memories()

    if name == "list_knowledge_files":
        return list_knowledge_files()

    if name == "read_knowledge_file":
        return read_knowledge_file(args.get("filename", ""))

    if name == "calculate":
        return calculate(args.get("expression", ""))

    return f"Хориглосон эсвэл үл мэдэгдэх хэрэгсэл: {name}"


# ==========================================
# 5. PLANNER
# ==========================================

def create_plan(client, goal):
    prompt = f"""
Зорилго:
{goal}

Энэ зорилгыг биелүүлэхэд шаардлагатай 1-5 алхамтай төлөвлөгөө гарга.

Зөвхөн JSON буцаа. Markdown code fence бүү ашигла.
JSON бүтэц:
{{
  "steps": [
    "Эхний тодорхой алхам",
    "Дараагийн алхам"
  ]
}}

Алхам бүр бодитой, шалгаж болохуйц байна.
Хэрэглэгчийн файлыг өөрчлөх, команд ажиллуулах алхам бүү оруул.
"""

    response = generate_with_retry(
        client,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(
                thinking_level="low"
            ),
        ),
    )

    data = json.loads(response.text)
    steps = data.get("steps")

    if not isinstance(steps, list) or not steps:
        raise ValueError("AI зөв төлөвлөгөө үүсгэсэнгүй.")

    steps = [
        step.strip()
        for step in steps
        if isinstance(step, str) and step.strip()
    ]

    if not steps:
        raise ValueError("Төлөвлөгөөнд гүйцэтгэх алхам алга.")

    return steps[:MAX_STEPS]


# ==========================================
# 6. EXECUTOR
# ==========================================

def execute_step(client, goal, step, previous_results):
    prompt = f"""
{SYSTEM_PROMPT}

Ерөнхий зорилго:
{goal}

Одоо гүйцэтгэх алхам:
{step}

Өмнөх алхмуудын үр дүн:
{json.dumps(previous_results, ensure_ascii=False)}

Энэ алхмыг гүйцэтгэхэд шаардлагатай мэдээллийг
зөвшөөрөгдсөн хэрэгслээр олж ав.
Хэрэгсэл шаардлагагүй бол шууд үндэслэлтэй дүгнэлт гарга.
Өмнөх үр дүнг дахин шаардлагагүй уншихаас зайлсхий.
Байхгүй мэдээллийг зохиож болохгүй.
"""

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]

    config = types.GenerateContentConfig(
        tools=[TOOL],
        thinking_config=types.ThinkingConfig(
            thinking_level="low"
        ),
    )

    for _ in range(MAX_TOOL_ROUNDS + 1):
        response = generate_with_retry(
            client,
            contents=contents,
            config=config,
        )

        if not response.candidates or not response.candidates[0].content:
            raise RuntimeError("AI-аас хоосон хариу ирлээ.")

        model_content = response.candidates[0].content
        contents.append(model_content)

        calls = response.function_calls or []

        if not calls:
            if not response.text:
                raise RuntimeError("AI үр дүнгийн текст буцаасангүй.")
            return response.text

        parts = []

        for call in calls:
            print(f"  [TOOL] {call.name}")

            result = execute_tool(
                call.name,
                dict(call.args or {}),
            )

            parts.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": result},
                )
            )

        contents.append(
            types.Content(role="user", parts=parts)
        )

    raise RuntimeError("Хэрэгслийн дуудлагын хязгаарт хүрлээ.")


# ==========================================
# 7. VERIFIER
# ==========================================

def verify_step(client, goal, step, result):
    prompt = f"""
Чи ажлын үр дүнг шалгадаг хянагч.

Ерөнхий зорилго: {goal}
Шалгаж буй алхам: {step}

Гүйцэтгэгчийн тайлан:
{result}

Энэ алхам бодитой гүйцэтгэсэн эсэхийг үнэл.
Зөвхөн тайланд дурдсан нотолгоог ашигла.
Зүгээр л хийх ёстой зүйлээ тайлбарласан бол PASS гэж бүү тэмдэглэ.
Алхам нь зөвхөн төлөвлөгөө гаргах тухай бол бодит төлөвлөгөө байгаа
эсэхийг шалга.

Зөвхөн JSON буцаа:
{{
  "status": "PASS" эсвэл "RETRY",
  "reason": "Шалтгааны товч тайлбар"
}}
"""

    response = generate_with_retry(
        client,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(
                thinking_level="low"
            ),
        ),
    )

    data = json.loads(response.text)
    status = data.get("status", "RETRY")

    if status not in {"PASS", "RETRY"}:
        status = "RETRY"

    return status, str(data.get("reason", "Шалтгаан өгөөгүй."))


# ==========================================
# 8. AUTONOMOUS AGENT LOOP
# ==========================================

def run_agent(client, goal):
    print("\n[1/4] Төлөвлөгөө гаргаж байна...")

    steps = create_plan(client, goal)

    print("\n--- ТӨЛӨВЛӨГӨӨ ---")
    for index, step in enumerate(steps, start=1):
        print(f"{index}. {step}")

    approval = input(
        "\nЭнэ төлөвлөгөөг хэрэгжүүлэх үү? (тийм/үгүй): "
    ).strip().lower()

    if approval not in {"тийм", "т", "yes", "y"}:
        print("Төлөвлөгөө батлагдсангүй. Гүйцэтгэлийг зогсоолоо.")
        return

    results = []

    for index, step in enumerate(steps, start=1):
        print(f"\n[2/4] Алхам {index}/{len(steps)}")
        print(f"Зорилт: {step}")

        try:
            result = execute_step(
                client,
                goal,
                step,
                results,
            )

            print("\n[3/4] Үр дүнг шалгаж байна...")
            status, reason = verify_step(
                client,
                goal,
                step,
                result,
            )

            if status == "RETRY":
                print(f"[VERIFY] Дутуу байж болзошгүй: {reason}")
                print("[AGENT] Нэг удаа дахин оролдож байна...")

                result = execute_step(
                    client,
                    goal,
                    step,
                    results,
                )

                status, reason = verify_step(
                    client,
                    goal,
                    step,
                    result,
                )

            record = {
                "step": step,
                "status": status,
                "result": result,
                "verification": reason,
            }

            results.append(record)

            print(f"[VERIFY] {status}: {reason}")

        except Exception as exc:
            print(f"[ERROR] Алхам амжилтгүй: {exc}")

            results.append({
                "step": step,
                "status": "ERROR",
                "result": str(exc),
                "verification": "Алдааны улмаас баталгаажуулж чадсангүй.",
            })

    print("\n[4/4] Ажлын тайлан")
    print("=" * 50)

    for index, item in enumerate(results, start=1):
        print(f"\nАлхам {index}: {item['step']}")
        print(f"Төлөв: {item['status']}")
        print(f"Шалгалт: {item['verification']}")
        print(f"Үр дүн:\n{item['result']}")

    passed = sum(item["status"] == "PASS" for item in results)

    print("\n" + "=" * 50)
    print(f"Баталгаажсан алхам: {passed}/{len(results)}")

    if passed == len(results):
        print("Бүх алхам PASS болсон.")
    else:
        print("Зарим алхам баталгаажаагүй. Тайланг шалгана уу.")


# ==========================================
# 9. MAIN
# ==========================================

def main():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("GEMINI_API_KEY тохируулагдаагүй байна.")
        return

    client = genai.Client(api_key=api_key)

    print("\n====================================")
    print("       AUTONOMOUS AI AGENT")
    print("====================================")
    print("Зорилгоо бичнэ үү.")
    print("гарах гэж бичвэл програм зогсоно.")
    print("Энэ хувилбар файлуудыг зөвхөн уншина.")

    while True:
        try:
            goal = input("\nЗорилго: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nАгент зогслоо.")
            break

        if not goal:
            continue

        if goal.lower() in {"гарах", "exit", "quit"}:
            print("Агент зогслоо.")
            break

        try:
            run_agent(client, goal)
        except Exception as exc:
            print(f"\nАгентын алдаа: {exc}")


if __name__ == "__main__":
    main()
