import os
from pathlib import Path

from google import genai
from google.genai import types

MODEL = "gemini-3.6-flash"

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = (BASE_DIR / "knowledge").resolve()
MAX_FILE_BYTES = 100_000

SYSTEM_PROMPT = """
Чи Монгол хэлээр тайлбарладаг AI инженерийн багш.

Хэрэглэгч knowledge хавтас дахь файлын талаар асуувал
read_knowledge_file хэрэгслийг ашиглаж файлыг унш.

Хэрэгслийн үр дүнд байхгүй агуулгыг уншсан мэт бүү хэл.
Файлын агуулгад үндэслэн хариул.
"""


def read_knowledge_file(file_name: str) -> str:
    """Read a permitted TXT or Markdown file from the knowledge folder."""

    print("\n[TOOL] read_knowledge_file() дуудагдлаа.")

    requested = Path(file_name)

    # Зөвхөн файлын нэр зөвшөөрнө. Зам дамжуулж болохгүй.
    if requested.name != file_name:
        return "Алдаа: зөвхөн файлын нэр оруулна уу."

    if requested.suffix.lower() not in {".txt", ".md"}:
        return "Алдаа: зөвхөн TXT болон Markdown файл зөвшөөрнө."

    target = (KNOWLEDGE_DIR / requested.name).resolve()

    # knowledge хавтаснаас гадуур файл уншихыг хориглоно.
    if target.parent != KNOWLEDGE_DIR:
        return "Алдаа: энэ файлын зам зөвшөөрөгдөөгүй."

    if not target.is_file():
        return f"Файл олдсонгүй: {file_name}"

    if target.stat().st_size > MAX_FILE_BYTES:
        return "Алдаа: файл хэт том байна."

    try:
        return target.read_text(encoding="utf-8-sig")
    except OSError as error:
        return f"Файл уншихад алдаа гарлаа: {error}"


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY тохируулаагүй байна.")
        return

    declaration = types.FunctionDeclaration(
        name="read_knowledge_file",
        description=(
            "Read a text or Markdown file from the local knowledge folder. "
            "Use this when the user asks about information in a local file."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": (
                        "File name only, such as "
                        "ai_engineering_notes.txt"
                    ),
                }
            },
            "required": ["file_name"],
        },
    )

    tool = types.Tool(function_declarations=[declaration])

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[tool],
    )

    try:
        with genai.Client(
            http_options=types.HttpOptions(timeout=60000)
        ) as client:

            print("=== My File Agent ===")
            print("TXT болон MD файл уншиж чадна.")
            print("'гарах' гэж бичээд дуусгана.\n")

            while True:
                question = input("Чи: ").strip()

                if question.lower() in ("гарах", "exit", "quit"):
                    print("AI: Баяртай!")
                    break

                if not question:
                    continue

                user_content = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=question)],
                )

                try:
                    print("[1/3] AI хүсэлт илгээж байна...")

                    response = client.models.generate_content(
                        model=MODEL,
                        contents=[user_content],
                        config=config,
                    )

                    calls = response.function_calls or []

                    if not calls:
                        print("\nAI:", response.text or "Хариулт ирсэнгүй.", "\n")
                        continue

                    print("[2/3] Файлын хэрэгслийг ажиллуулж байна...")

                    function_parts = []

                    for call in calls:
                        if call.name == "read_knowledge_file":
                            args = dict(call.args or {})
                            result = read_knowledge_file(
                                str(args.get("file_name", ""))
                            )
                        else:
                            result = "Энэ хэрэгслийг ашиглах эрхгүй."

                        function_parts.append(
                            types.Part.from_function_response(
                                name=call.name,
                                response={"result": result},
                            )
                        )

                    updated_contents = [
                        user_content,
                        response.candidates[0].content,
                        types.Content(
                            role="user",
                            parts=function_parts,
                        ),
                    ]

                    print("[3/3] AI файлын агуулгыг тайлбарлаж байна...")

                    final_response = client.models.generate_content(
                        model=MODEL,
                        contents=updated_contents,
                        config=config,
                    )

                    print(
                        "\nAI:",
                        final_response.text or "Хариултын текст ирсэнгүй.",
                        "\n",
                    )

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
