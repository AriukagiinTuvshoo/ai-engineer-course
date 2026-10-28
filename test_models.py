import time

from google import genai
from google.genai import types

CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
]

PROMPT = "Монгол хэлээр зөвхөн 'Сайн уу' гэж хариул."


def try_model(client, model, use_thinking):
    config = None
    if use_thinking:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low")
        )
    start = time.time()
    try:
        r = client.models.generate_content(
            model=model, contents=PROMPT, config=config
        )
        return f"OK {time.time() - start:.1f}с -> {r.text!r}"
    except Exception as e:
        msg = str(e).replace("\n", " ")[:110]
        return f"АЛДАА {time.time() - start:.1f}с -> {type(e).__name__}: {msg}"


def main():
    with genai.Client(http_options=types.HttpOptions(timeout=30000)) as client:
        for model in CANDIDATES:
            print(f"\n{model}")
            print("  thinking-гүй :", try_model(client, model, False))
            print("  thinking low :", try_model(client, model, True))


if __name__ == "__main__":
    main()
