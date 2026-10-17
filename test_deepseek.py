import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

key = os.getenv("AMD_API_KEY")
if not key:
    raise ValueError("AMD_API_KEY missing from .env")

client = OpenAI(
    api_key=key,
    base_url="https://developer.amd.com.cn/radeon/api/v1"
)

response = client.chat.completions.create(
    model="DeepSeek-V4-Flash",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: API connection successful"
        }
    ],
    max_tokens=50
)

print(response.choices[0].message.content)
