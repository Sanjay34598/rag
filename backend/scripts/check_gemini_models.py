import os
from dotenv import load_dotenv
from pathlib import Path
from groq import Groq

env_path = Path(".env") if Path(".env").exists() else Path("../.env")
load_dotenv(env_path)

key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=key)
res = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello, answer in JSON format: {\"answer\": \"Hi\"}"}],
    model="openai/gpt-oss-20b",
    response_format={"type": "json_object"}
)
print("Groq Model 'openai/gpt-oss-20b' -> Output:", res.choices[0].message.content)

