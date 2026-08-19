# Deepnote Deployment Guide for Voice RAG FastAPI Backend

This guide details how to deploy and run the FastAPI backend for Voice RAG on **Deepnote** and connect it to your **Vercel** frontend.

---

## 1. Getting the Repository into Deepnote

1. Log in to your [Deepnote](https://deepnote.com) account.
2. Create or open your Deepnote project workspace.
3. Clone or upload this repository into the workspace environment.

---

## 2. Required Directory Structure

Ensure the backend folder is located at:
```text
/datasets/_deepnote_work/rag/backend
```
If your repository path is slightly different, ensure `sys.path` in the startup script points to your `backend` root folder containing `app/main.py`.

---

## 3. Install Dependencies

In a Deepnote terminal or Python notebook cell, run:

```bash
pip install -r /datasets/_deepnote_work/rag/backend/requirements.txt
```

---

## 4. Required Environment Variables

Configure the following environment variables in your Deepnote Project Settings (Environment Variables panel) or set them in python before launching the server:

| Environment Variable | Recommended Value / Purpose |
|----------------------|-----------------------------|
| `GROQ_API_KEY` | Your secret Groq API key (Backend-only) |
| `SARVAM_API_KEY` | Your secret Sarvam STT API key (Backend-only) |
| `LLM_MODEL` | `llama-3.1-8b-instant` |
| `LLM_MODE` | `real` |
| `LLM_PROVIDER` | `groq` |
| `CORS_ORIGINS` | `https://voice-b0064qrq6-sanjays-projects-f2a71297.vercel.app,http://localhost:5173,http://localhost:3000` |

> [!IMPORTANT]
> Never put actual API keys in public git repositories. Set them securely inside Deepnote Environment Settings.

---

## 5. Starting FastAPI in Deepnote Notebook

Because Deepnote notebook cells run inside an existing `asyncio` event loop, running `uvicorn.run()` directly will throw a `RuntimeError: asyncio.run() cannot be called from a running event loop`.

Use the following thread-based startup code inside a Deepnote notebook cell:

```python
import os
import sys
import threading
import uvicorn

# Ensure environment variables are loaded
os.environ["LLM_MODEL"] = "llama-3.1-8b-instant"
os.environ["LLM_MODE"] = "real"
os.environ["LLM_PROVIDER"] = "groq"
os.environ["CORS_ORIGINS"] = "https://voice-b0064qrq6-sanjays-projects-f2a71297.vercel.app,http://localhost:5173,http://localhost:3000"
# os.environ["GROQ_API_KEY"] = "your_actual_key"
# os.environ["SARVAM_API_KEY"] = "your_actual_key"

BACKEND = "/datasets/_deepnote_work/rag/backend"

if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

def start_api():
    config = uvicorn.Config(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )

    server = uvicorn.Server(config)
    server.run()

api_thread = threading.Thread(
    target=start_api,
    daemon=True
)

api_thread.start()

print("Voice RAG API starting on port 8080...")
```

---

## 6. Local Health Verification in Deepnote

In a separate notebook cell or terminal, verify the server is running:

```bash
curl http://127.0.0.1:8080/health
```

Expected JSON response:
```json
{
    "status": "ok",
    "version": "0.4.1",
    "rag_ready": true
}
```

---

## 7. Test CORS Endpoint

```bash
curl http://127.0.0.1:8080/cors-test
```

Expected JSON response:
```json
{
    "status": "ok",
    "message": "CORS is working"
}
```

---

## 8. Exposing Port 8080 Publicly in Deepnote

1. In Deepnote, navigate to **Project Settings** / **Environment** / **Services & Ingress** (or Port Forwarding).
2. Set port `8080` as exposed to public HTTPS access.
3. Deepnote will generate a public HTTPS URL for port 8080.

---

## 9. Determining your Public Deepnote URL

Your public Deepnote backend URL will look like:
```text
https://eeabc223-3d6b-4a65-83b8-b48ff39aceac.sandbox-prod-e1f.deepnoteproject.com
```

> [!WARNING]
> Do NOT append `/health` or a trailing slash `/` to your base API URL when configuring the frontend!

---

## 10. Configuring Vercel `VITE_API_URL`

1. Go to your [Vercel Dashboard](https://vercel.com).
2. Select your project: `voice-b0064qrq6-sanjays-projects-f2a71297` (or your active Vercel frontend project).
3. Go to **Settings** -> **Environment Variables**.
4. Add or update `VITE_API_URL`:
   - Key: `VITE_API_URL`
   - Value: `https://eeabc223-3d6b-4a65-83b8-b48ff39aceac.sandbox-prod-e1f.deepnoteproject.com`
   - Environments: `Production`, `Preview`, `Development`

---

## 11. Redeploying Vercel

Vite bakes environment variables into static assets during build time.
After setting `VITE_API_URL` in Vercel:
1. Go to **Deployments** tab in Vercel.
2. Click **Redeploy** on the latest deployment (or run `git push`).

---

## 12. Troubleshooting CORS and API Connections

If the Vercel frontend displays "Failed to connect to Voice RAG API endpoint.":

1. **Verify `CORS_ORIGINS`**: Ensure Deepnote backend environment variable `CORS_ORIGINS` includes exact Vercel frontend domain:
   `https://voice-b0064qrq6-sanjays-projects-f2a71297.vercel.app`
2. **Verify public URL works in browser**: Open `https://eeabc223-3d6b-4a65-83b8-b48ff39aceac.sandbox-prod-e1f.deepnoteproject.com/cors-test` in your browser.
3. **Check Browser Console / DevTools Network Tab**:
   - Check Preflight (`OPTIONS`) request response headers.
   - Look for `access-control-allow-origin: https://voice-b0064qrq6-sanjays-projects-f2a71297.vercel.app`.
4. **Ensure `VITE_API_URL` does NOT contain `/health`**:
   - Correct: `https://eeabc223-3d6b-4a65-83b8-b48ff39aceac.sandbox-prod-e1f.deepnoteproject.com`
   - Incorrect: `https://eeabc223-3d6b-4a65-83b8-b48ff39aceac.sandbox-prod-e1f.deepnoteproject.com/health`
