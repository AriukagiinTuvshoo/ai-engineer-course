import os
import sqlite3
import time

from google import genai
from google.genai import errors, types

# Таны туршилтаар ажилласан загварууд (gemini-3.8 / 3.7 нь 504 өгч байсан)
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_FALLBACK", "gemini-3.5-flash,gemini-3.5-flash-lite"
    ).split(",")
    if m.strip()
]
DB_PATH = "ai_memory.db"
RETRY_DELAYS = [1, 2]  # секунд. Загвар тус бүр хамгийн ихдээ 3 удаа оролдоно
REQUEST_TIMEOUT_MS = 30000  # 30 секунд

SYSTEM_PROMPT = """
Чи Монгол хэлээр ярьдаг AI инженерийн багш.
Хэрэглэгч өмнө хадгалсан мэдээллээ асуувал
read_saved_memories хэрэгслийг ашигла.
Өгөгдлийн санд байхгүй мэдээллийг зохиож болохгүй.
"""


# ---------- Өгөгдлийн сан ----------

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
        rows = conn.execute(
            "SELECT key, value FROM memories ORDER BY key"
        ).fetchall()
    return dict(rows)


def read_saved_memories():
    print("\n[TOOL] Санах ойн функц ажиллаж байна.")
    memories = get_memories()
    if not memories:
        return "Хадгалсан мэдээлэл алга."
    return "\n".join(f"{key}: {value}" for key, value in memories.items())


def save_memory(key, value):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO memories (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
        """, (key, value))


def delete_memory(key):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        return cursor.rowcount > 0


# ---------- API дуудлага (retry + fallback) ----------

def is_retryable(error):
    if isinstance(error, errors.ServerError):  # 5xx (504 орно)
        return True
    if isinstance(error, errors.ClientError) and getattr(error, "code", None) == 429:
        return True
    return False


def generate_with_retry(client, **kwargs):
    """Үндсэн загварыг retry-тай дуудаж, бүтэлгүйтвэл нөөц загвар руу шилжинэ."""
    primary = kwargs.pop("model", MODEL)
    models_to_try = [primary] + [m for m in FALLBACK_MODELS if m != primary]
    last_error = None

    for model in models_to_try:
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                if model != primary and attempt == 0:
                    print(f"[!] Нөөц загвар руу шилжиж байна: {model}")
                return client.models.generate_content(model=model, **kwargs)
            except Exception as error:
                last_error = error
                if not is_retryable(error):
                    raise
                if attempt == len(RETRY_DELAYS):
                    break  # энэ загвар дууслаа, дараагийн загвар руу
                delay = RETRY_DELAYS[attempt]
                code = getattr(error, "code", "?")
                print(f"[!] {model}: алдаа ({code}), {delay}с дараа дахин оролдоно "
                      f"({attempt + 1}/{len(RETRY_DELAYS)})...")
                time.sleep(delay)

    raise last_error


# ---------- Оношилгоо ----------

def run_diagnostics(client, config):
    print("\n=== Оношилгоо ===")

    # 1. API key
    ok = bool(os.environ.get("GEMINI_API_KEY"))
    print(f"1. GEMINI_API_KEY: {'OK' if ok else 'ТОХИРУУЛААГҮЙ'}")

    # 2. Өгөгдлийн сан
    try:
        count = len(get_memories())
        print(f"2. Өгөгдлийн сан: OK ({count} бичлэг)")
    except Exception as e:
        print(f"2. Өгөгдлийн сан: АЛДАА - {e}")

    # 3. Санах ойн хэрэгсэл (AI-гүйгээр)
    try:
        start = time.time()
        read_saved_memories()
        print(f"3. Хэрэгсэл: OK ({time.time() - start:.3f}с)")
    except Exception as e:
        print(f"3. Хэрэгсэл: АЛДАА - {e}")

    # 4. Загвар жагсаалтад байгаа эсэх (ашиглах боломжтой гэсэн үг биш, 5-р алхмаар батална)
    for m in [MODEL] + FALLBACK_MODELS:
        try:
            client.models.get(model=m)
            print(f"4. Загвар '{m}': жагсаалтад байна")
        except Exception as e:
            print(f"4. Загвар '{m}': АЛДАА - {type(e).__name__}: {e}")

    # 5. Энгийн хүсэлт (tool-гүй) хэр удаан хариулж байна
    try:
        start = time.time()
        simple_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low")
        )
        r = client.models.generate_content(
            model=MODEL, contents="Сайн уу гэж нэг үгээр хариул.",
            config=simple_config,
        )
        print(f"5. Энгийн хүсэлт: OK ({time.time() - start:.1f}с) -> {r.text!r}")
    except Exception as e:
        print(f"5. Энгийн хүсэлт: АЛДАА - {type(e).__name__}: {e}")

    # 6. Tool-той хүсэлт
    try:
        start = time.time()
        r = client.models.generate_content(
            model=MODEL, contents="Миний нэрийг санах ойгоос хараад хэл.",
            config=config,
        )
        called = [c.name for c in (r.function_calls or [])]
        print(f"6. Tool-той хүсэлт: OK ({time.time() - start:.1f}с), "
              f"дуудсан хэрэгсэл: {called or 'байхгүй'}")
    except Exception as e:
        print(f"6. Tool-той хүсэлт: АЛДАА - {type(e).__name__}: {e}")

    print("=================\n")


# ---------- Үндсэн програм ----------

def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY тохируулаагүй байна.")
        return

    init_db()

    tool_definition = types.FunctionDeclaration(
        name="read_saved_memories",
        description="Read the user's saved name, goals and profile.",
        parameters_json_schema={"type": "object", "properties": {}},
    )
    tool = types.Tool(function_declarations=[tool_definition])

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[tool],
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )

    history = []

    try:
        with genai.Client(
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
        ) as client:

            print("=== My AI Agent ===")
            print(f"Загвар: {MODEL}  |  Нөөц: {', '.join(FALLBACK_MODELS) or 'байхгүй'}")
            print("/санах нэр Түвшөө")
            print("/санах зорилго AI инженер болох")
            print("/ой")
            print("/мартах нэр")
            print("/шалгах   (оношилгоо)")
            print("шинэ яриа")
            print("гарах\n")

            while True:
                message = input("Чи: ").strip()

                if not message:
                    continue

                command = message.lower()

                if command in ("гарах", "exit", "quit"):
                    print("AI: Баяртай!")
                    break

                if command == "шинэ яриа":
                    history.clear()
                    print("Ярианы түүх цэвэрлэгдлээ.\n")
                    continue

                if command == "/ой":
                    memories = get_memories()
                    print(memories or "Хадгалсан мэдээлэл алга.")
                    print()
                    continue

                if command == "/шалгах":
                    run_diagnostics(client, config)
                    continue

                if command.startswith("/санах "):
                    parts = message.split(maxsplit=2)
                    if len(parts) != 3:
                        print("Жишээ: /санах нэр Түвшөө\n")
                        continue
                    save_memory(parts[1].lower(), parts[2])
                    history.clear()
                    print("Мэдээлэл хадгаллаа.\n")
                    continue

                if command.startswith("/мартах "):
                    parts = message.split(maxsplit=1)
                    if len(parts) != 2:
                        print("Жишээ: /мартах нэр\n")
                        continue
                    if delete_memory(parts[1].lower()):
                        history.clear()
                        print("Мэдээлэл устгалаа.\n")
                    else:
                        print("Тийм мэдээлэл олдсонгүй.\n")
                    continue

                user_content = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)],
                )
                request_contents = history + [user_content]

                try:
                    print("[1/3] AI-д хүсэлт илгээж байна...")
                    start = time.time()

                    response = generate_with_retry(
                        client,
                        model=MODEL,
                        contents=request_contents,
                        config=config,
                    )

                    calls = response.function_calls or []

                    if calls:
                        model_content = response.candidates[0].content
                        updated_history = request_contents + [model_content]

                        for call in calls:
                            if call.name != "read_saved_memories":
                                result = "Зөвшөөрөөгүй хэрэгсэл."
                            else:
                                print("[2/3] Python хэрэгслийг ажиллуулж байна...")
                                result = read_saved_memories()

                            tool_part = types.Part.from_function_response(
                                name=call.name,
                                response={"result": result},
                            )
                            updated_history.append(
                                types.Content(role="user", parts=[tool_part])
                            )

                        print("[3/3] AI эцсийн хариултаа боловсруулж байна...")

                        final_response = generate_with_retry(
                            client,
                            model=MODEL,
                            contents=updated_history,
                            config=config,
                        )

                        answer = final_response.text or "Хариултын текст ирсэнгүй."
                        history = updated_history + [
                            final_response.candidates[0].content
                        ]
                    else:
                        answer = response.text or "Текст хариулт ирсэнгүй."
                        history = request_contents + [
                            response.candidates[0].content
                        ]

                    print(f"\nAI: {answer}")
                    print(f"({time.time() - start:.1f}с)\n")

                except Exception as error:
                    print("\nХүсэлт амжилтгүй боллоо.")
                    print("Алдааны төрөл:", type(error).__name__)
                    print("Дэлгэрэнгүй:", error)

    except KeyboardInterrupt:
        print("\nПрограммаас гарлаа.")
    except Exception as error:
        print("Эхлүүлэх алдаа:", type(error).__name__, error)


if __name__ == "__main__":
    main()