import os
import sqlite3
from contextlib import closing

from google import genai
from google.genai import types

MODEL = "gemini-3.8-flash"
DB_PATH = "ai_memory.db"

ALLOWED_KEYS = {"нэр", "зорилго", "хэл", "төсөл"}

SYSTEM_PROMPT = """
Чи Монгол хэлээр ойлгомжтой ярьдаг AI багш.
Програмчлал болон AI инженерчлэлийг эхнээс нь заа.
Хэрэглэгчийн хадгалсан мэдээллийг тохирох үед ашигла.
Мэдэхгүй зүйлээ зохиож хэлэхгүй.
"""


def connect_db():
    return sqlite3.connect(DB_PATH)


def initialize_db():
    with closing(connect_db()) as conn:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)


def save_memory(key, value):
    with closing(connect_db()) as conn:
        with conn:
            conn.execute("""
                INSERT INTO memories (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
            """, (key, value))


def load_memories():
    with closing(connect_db()) as conn:
        rows = conn.execute(
            "SELECT key, value FROM memories ORDER BY key"
        ).fetchall()

    return dict(rows)


def delete_memory(key):
    with closing(connect_db()) as conn:
        with conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE key = ?",
                (key,)
            )
            return cursor.rowcount > 0


def create_chat(client):
    return client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        ),
    )


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY тохируулаагүй байна.")
        return

    initialize_db()

    try:
        with genai.Client(
            http_options=types.HttpOptions(timeout=60000)
        ) as client:
            chat = create_chat(client)

            print("=== My AI Assistant ===")
            print("Санах нэр Түвшөө")
            print("Санах зорилго AI инженер болох")
            print("Санах ой")
            print("Мартах нэр")
            print("Шинэ яриа")
            print("Гарах\n")

            while True:
                message = input("Чи: ").strip()
                command = message.lower()

                if command in ("гарах", "exit", "quit"):
                    print("AI: Баяртай!")
                    break

                if command == "шинэ яриа":
                    chat = create_chat(client)
                    print("AI: Шинэ яриа эхэллээ.\n")
                    continue

                if command == "санах ой":
                    memories = load_memories()

                    if memories:
                        print("\nХадгалсан мэдээлэл:")
                        for key, value in memories.items():
                            print(f"- {key}: {value}")
                    else:
                        print("Одоогоор хадгалсан мэдээлэл алга.")

                    print()
                    continue

                if command.startswith("санах "):
                    parts = message.split(maxsplit=2)

                    if (
                        len(parts) != 3
                        or parts[1].lower() not in ALLOWED_KEYS
                        or not parts[2].strip()
                    ):
                        print(
                            "Жишээ: Санах нэр Түвшөө\n"
                            "Зөв түлхүүр: нэр, зорилго, хэл, төсөл\n"
                        )
                        continue

                    key = parts[1].lower()
                    value = parts[2].strip()

                    save_memory(key, value)
                    chat = create_chat(client)

                    print(f"AI: '{key}' мэдээллийг өгөгдлийн санд хадгаллаа.\n")
                    continue

                if command.startswith("мартах "):
                    parts = message.split(maxsplit=1)
                    key = parts[1].strip().lower()

                    if key not in ALLOWED_KEYS:
                        print("Нэр, зорилго, хэл эсвэл төсөл гэж бичээрэй.\n")
                        continue

                    if delete_memory(key):
                        print(f"AI: '{key}' мэдээллийг устгалаа.\n")
                    else:
                        print("Тийм мэдээлэл хадгалагдаагүй байна.\n")

                    chat = create_chat(client)
                    continue

                if not message:
                    continue

                memories = load_memories()

                if memories:
                    memory_text = "\n".join(
                        f"- {key}: {value}"
                        for key, value in memories.items()
                    )
                else:
                    memory_text = "Хадгалсан мэдээлэл алга."

                prompt = (
                    "Хэрэглэгчийн хадгалсан мэдээлэл:\n"
                    f"{memory_text}\n\n"
                    "Эдгээр мэдээллийг асуултад тохирох үед ашигла.\n"
                    f"Хэрэглэгчийн асуулт: {message}"
                )

                try:
                    response = chat.send_message(prompt)
                    print(f"\nAI: {response.text}\n")
                except Exception as error:
                    print("AI хүсэлт амжилтгүй боллоо.")
                    print("Алдааны төрөл:", type(error).__name__)
                    print("Дэлгэрэнгүй:", error)

    except Exception as error:
        print("Программ эхлүүлэхэд алдаа гарлаа.")
        print("Алдааны төрөл:", type(error).__name__)
        print("Дэлгэрэнгүй:", error)


if __name__ == "__main__":
    main()
