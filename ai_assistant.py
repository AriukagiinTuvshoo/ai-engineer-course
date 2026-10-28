
import os
import sqlite3
import time
from pathlib import Path

from google import genai
from google.genai import errors, types


# =========================
# 1. ҮНДСЭН ТОХИРГОО
# =========================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "ai_memory.db"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

PRIMARY_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)

FALLBACK_MODELS = [
    model.strip()
    for model in os.environ.get(
        "GEMINI_FALLBACK",
        "gemini-3.5-flash,gemini-3.5-flash-lite",
    ).split(",")
    if model.strip()
]

RETRY_DELAYS = [1, 2]
MAX_FILE_BYTES = 100_000
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = """
Чи бол хэрэглэгчийн хувийн AI туслах бөгөөд AI инженерчлэлийн багш.

Хэрэглэгчтэй монгол хэлээр ойлгомжтой, бодитой, найрсаг харилц.
Кодын талаар тайлбарлахдаа эхлэгч хүнд ойлгомжтой жишээ ашигла.

Хэрэглэгчийн хадгалсан мэдээллийг мэдэх шаардлагатай үед
read_saved_memories хэрэгслийг ашигла.
Орон нутгийн мэдлэгийн файл унших шаардлагатай үед
read_knowledge_file хэрэгслийг ашигла.
Хоёр эх сурвалж хоёулаа хэрэгтэй бол хоёуланг нь ашигла.

Уншаагүй файл, шалгаагүй өгөгдлийг уншсан мэт бүү хэл.
Санах ойд байхгүй мэдээллийг зохиож бүү нэм.
Хэрэглэгч санах ойд мэдээлэл хадгал гэж шууд хүсээгүй бол
зөвхөн ярианд дурдсан зүйлээс автоматаар байнгын санах ойд хадгалж болохгүй.

Хариултаа шууд, хэрэгтэй байдлаар өг.
"""


# =========================
# 2. SQLITE САНАХ ОЙ
# =========================

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def get_memories():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            "SELECT key, value FROM memories ORDER BY key"
        ).fetchall()


def read_saved_memories():
    memories = get_memories()

    if not memories:
        return "Санах ой одоогоор хоосон байна."

    return "\n".join(
        f"{key}: {value}" for key, value in memories
    )


def save_memory(key, value):
    key = key.strip()
    value = value.strip()

    if not key or not value:
        raise ValueError("Түлхүүр болон утга хоосон байж болохгүй.")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO memories (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, value),
        )


def delete_memory(key):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "DELETE FROM memories WHERE key = ?",
            (key.strip(),),
        )
        return cursor.rowcount > 0


# =========================
# 3. ФАЙЛ УНШИХ ХЭРЭГСЭЛ
# =========================

def read_knowledge_file(filename):
    """
    knowledge хавтас доторх .txt эсвэл .md файлыг уншина.
    Файлын нэрээр л хандана; зам зааж хавтаснаас гарахыг хориглоно.
    """

    if not isinstance(filename, str) or not filename.strip():
        return "Алдаа: файлын нэр шаардлагатай."

    filename = filename.strip()

    # Зөвхөн файлын нэр зөвшөөрнө.
    if (
        Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or filename in {".", ".."}
    ):
        return "Алдаа: зөвхөн knowledge хавтас дахь файлын нэрийг өгнө."

    file_path = (KNOWLEDGE_DIR / filename).resolve()

    try:
        file_path.relative_to(KNOWLEDGE_DIR.resolve())
    except ValueError:
        return "Алдаа: зөвшөөрөөгүй файлын зам."

    if file_path.suffix.lower() not in {".txt", ".md"}:
        return "Алдаа: зөвхөн .txt болон .md файлыг уншина."

    if not file_path.is_file():
        return f"Алдаа: '{filename}' файл олдсонгүй."

    try:
        if file_path.stat().st_size > MAX_FILE_BYTES:
            return "Алдаа: файл хэт том байна. 100 KB-аас бага файл ашиглана уу."

        return file_path.read_text(encoding="utf-8-sig")

    except UnicodeDecodeError:
        return "Алдаа: файлын кодчиллыг UTF-8 болгоно уу."
    except OSError as exc:
        return f"Файл уншихад алдаа гарлаа: {exc}"


# =========================
# 4. AI ХЭРЭГСЛИЙН ТОДОРХОЙЛОЛТ
# =========================

TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="read_saved_memories",
            description=(
                "Хэрэглэгчийн SQLite санах ойд хадгалсан "
                "мэдээллийг уншина. Аргумент шаардлагагүй."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
        types.FunctionDeclaration(
            name="read_knowledge_file",
            description=(
                "knowledge хавтас дахь .txt эсвэл .md файлыг уншина. "
                "Бүтэн зам биш, зөвхөн файлын нэр өг."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "filename": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Файлын нэр. Жишээ: ai_engineering_notes.txt"
                        ),
                    )
                },
                required=["filename"],
            ),
        ),
    ]
)


# =========================
# 5. RETRY + FALLBACK
# =========================

def is_retryable(error):
    if isinstance(error, errors.ServerError):
        return True

    if isinstance(error, errors.ClientError):
        return getattr(error, "code", None) == 429

    return False


def generate_with_retry(client, **kwargs):
    primary_model = kwargs.pop("model", PRIMARY_MODEL)

    models = list(dict.fromkeys(
        [primary_model] + FALLBACK_MODELS
    ))

    last_error = None

    for model in models:
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                print(
                    f"\n[AI] Загвар: {model} | "
                    f"Оролдлого: {attempt + 1}/"
                    f"{len(RETRY_DELAYS) + 1}"
                )

                response = client.models.generate_content(
                    model=model,
                    **kwargs,
                )

                print(f"[AI] Амжилттай: {model}")
                return response

            except Exception as exc:
                last_error = exc

                if not is_retryable(exc):
                    # Загварын нэр хүчингүй зэрэг бусад алдааг
                    # нуухгүй, хэрэглэгчид ил тод харуулна.
                    raise

                if attempt < len(RETRY_DELAYS):
                    delay = RETRY_DELAYS[attempt]
                    print(
                        f"[AI] Түр зуурын алдаа. "
                        f"{delay} секундийн дараа дахин оролдоно."
                    )
                    time.sleep(delay)
                else:
                    print(f"[AI] {model} загвар амжилтгүй.")

    raise RuntimeError(
        f"Бүх загвар амжилтгүй боллоо: {last_error}"
    )


# =========================
# 6. AI ХАРИУЛТ БА TOOL CALLING
# =========================

def execute_tool(name, args):
    if name == "read_saved_memories":
        return read_saved_memories()

    if name == "read_knowledge_file":
        filename = args.get("filename", "")
        return read_knowledge_file(filename)

    return f"Алдаа: зөвшөөрөөгүй хэрэгслийн нэр: {name}"


def ask_ai(client, history, user_message):
    history.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[TOOL],
        thinking_config=types.ThinkingConfig(
            thinking_level="low"
        ),
    )

    for _ in range(MAX_TOOL_ROUNDS + 1):
        response = generate_with_retry(
            client,
            model=PRIMARY_MODEL,
            contents=history,
            config=config,
        )

        candidate = (
            response.candidates[0]
            if response.candidates
            else None
        )

        if candidate is None or candidate.content is None:
            raise RuntimeError("AI-аас хоосон хариу ирлээ.")

        model_content = candidate.content
        history.append(model_content)

        function_calls = response.function_calls or []

        if not function_calls:
            answer = response.text

            if not answer:
                raise RuntimeError("AI текстэн хариу үүсгэсэнгүй.")

            return answer

        function_responses = []

        for call in function_calls:
            print(f"[TOOL] {call.name}() дуудагдлаа.")

            args = dict(call.args or {})
            result = execute_tool(call.name, args)

            function_responses.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": result},
                )
            )

        history.append(
            types.Content(
                role="user",
                parts=function_responses,
            )
        )

    raise RuntimeError(
        "AI хэт олон удаа хэрэгсэл дуудаж байна. "
        "Асуултаа илүү тодорхой болгоод дахин оролдоно уу."
    )


# =========================
# 7. ОНОШИЛГОО
# =========================

def run_diagnostics(client):
    print("\n========== AI ОНОШИЛГОО ==========")

    if os.environ.get("GEMINI_API_KEY"):
        print("[OK] GEMINI_API_KEY тохируулагдсан.")
    else:
        print("[FAIL] GEMINI_API_KEY олдсонгүй.")
        print(
            "PowerShell дээр $env:GEMINI_API_KEY тохируулна уу."
        )
        return

    try:
        init_db()
        print(f"[OK] SQLite сан: {DB_PATH}")
        print(f"[OK] Санах ойд {len(get_memories())} бичлэг байна.")
    except Exception as exc:
        print(f"[FAIL] SQLite: {exc}")
        return

    if KNOWLEDGE_DIR.is_dir():
        print(f"[OK] knowledge хавтас: {KNOWLEDGE_DIR}")
    else:
        print("[WARN] knowledge хавтас олдсонгүй.")

    try:
        response = generate_with_retry(
            client,
            model=PRIMARY_MODEL,
            contents="Зөвхөн OK гэж хариул.",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                )
            ),
        )
        print(f"[OK] Gemini холболт ажиллаж байна: {response.text}")
    except Exception as exc:
        print(f"[FAIL] Gemini холболт: {exc}")

    print("==================================\n")


# =========================
# 8. ТУСЛАХЫН ҮНДСЭН ЦИКЛ
# =========================

def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("Алдаа: GEMINI_API_KEY тохируулагдаагүй байна.")
        print(
            "PowerShell дээр API түлхүүрээ тохируулаад "
            "дахин ажиллуулна уу."
        )
        return

    try:
        init_db()
        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )
    except Exception as exc:
        print(f"Эхлүүлэхэд алдаа гарлаа: {exc}")
        return

    history = []

    print("\n======================================")
    print("       ХУВИЙН AI ТУСЛАХ")
    print("======================================")
    print(f"Үндсэн загвар: {PRIMARY_MODEL}")
    print(f"Санах ой: {DB_PATH}")
    print("\nКомандууд:")
    print("  /санах <түлхүүр> <утга> - мэдээлэл хадгалах")
    print("  /ой                     - санах ой харах")
    print("  /мартах <түлхүүр>       - мэдээлэл устгах")
    print("  /шалгах                 - систем шалгах")
    print("  шинэ яриа               - ярианы түүх цэвэрлэх")
    print("  гарах                   - програмаас гарах")
    print("--------------------------------------")

    while True:
        try:
            message = input("\nТа: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAI туслах зогслоо.")
            break

        if not message:
            continue

        if message.lower() in {"гарах", "exit", "quit"}:
            print("AI туслах зогслоо.")
            break

        if message == "шинэ яриа":
            history.clear()
            print("Одоогийн ярианы түүх цэвэрлэгдлээ.")
            print("SQLite санах ой хэвээр үлдэнэ.")
            continue

        if message == "/ой":
            print("\n--- САНАХ ОЙ ---")
            print(read_saved_memories())
            continue

        if message.startswith("/санах "):
            parts = message.split(maxsplit=2)

            if len(parts) != 3:
                print("Жишээ: /санах зорилго AI инженер болох")
                continue

            try:
                save_memory(parts[1], parts[2])
                print(f"Хадгаллаа: {parts[1]} = {parts[2]}")
            except Exception as exc:
                print(f"Хадгалахад алдаа гарлаа: {exc}")

            continue

        if message.startswith("/мартах "):
            parts = message.split(maxsplit=1)

            if len(parts) != 2:
                print("Жишээ: /мартах зорилго")
                continue

            if delete_memory(parts[1]):
                print(f"Устгалаа: {parts[1]}")
            else:
                print("Тэр түлхүүр санах ойд байхгүй байна.")

            continue

        if message == "/шалгах":
            run_diagnostics(client)
            continue

        try:
            answer = ask_ai(client, history, message)
            print(f"\nAI: {answer}")
        except Exception as exc:
            print(f"\nАлдаа гарлаа: {exc}")
            print(
                "Алдаа арилсны дараа дахин оролдож болно. "
                "Оношилгоонд /шалгах командыг ашигла."
            )


if __name__ == "__main__":
    main()
