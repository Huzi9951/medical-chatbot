<p align="center">✦ &nbsp; HUZI9951 &nbsp; ✦</p>

# Local medical information chat

A small FastAPI front for a local [vLLM](https://docs.vllm.ai/) server running Qwen2.5-Instruct. The page posts the conversation to `/chat`. The server adds a system prompt and round-robins across one or more vLLM URLs.

This is general information, not a diagnosis, and not a substitute for a doctor. The page says that, and the system prompt says it again.

## Run

Start vLLM with the model you actually have, then point the app at it:

```bash
pip install fastapi uvicorn openai pydantic
```

```bash
set VLLM_URLS=http://localhost:8000/v1
set VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
uvicorn backend_med:app --port 8080
```

vLLM stays on port 8000. This app listens on 8080 so the two do not collide. If you change the port, change `API_BASE` at the top of the script in `medical_chat.html`.

Open `medical_chat.html`. The page calls `http://127.0.0.1:8080/chat`.

`GET /health` returns how many backends are configured.

The previous page pointed at a temporary Cloudflare tunnel URL. That tunnel is not part of the project anymore.
