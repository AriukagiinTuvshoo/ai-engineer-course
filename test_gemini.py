from google import genai
from google.genai import types

try:
    client = genai.Client(
        http_options=types.HttpOptions(timeout=30000)
    )

    print("AI-д хүсэлт илгээж байна...")

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Монгол хэлээр зөвхөн 'Сайн уу, Түвшөө!' гэж хариул."
    )

    print("AI-ийн хариулт:", response.text)

except Exception as error:
    print("Алдааны төрөл:", type(error).__name__)
    print("Дэлгэрэнгүй:", error)
