import os
import requests
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(".env") if Path(".env").exists() else Path("../.env")
load_dotenv(env_path)

key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")

for model in ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-flash-latest", "gemini-1.5-flash"]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": "Hello, answer in JSON format: {\"answer\": \"Hi\"}"}]}]
    }
    res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    print(f"Model '{model}' -> HTTP {res.status_code}")
    if res.status_code == 200:
        print("  Response:", res.json()["candidates"][0]["content"]["parts"][0]["text"])
