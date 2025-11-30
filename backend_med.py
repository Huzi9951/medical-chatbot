# backend_med.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal
from openai import OpenAI
import itertools
import os

# ---------- CONFIG ----------
# vLLM OpenAI endpoint(s)
VLLM_URLS = os.getenv("VLLM_URLS", "http://localhost:8000/v1").split(",")

# Model name/path must match what vLLM was started with
VLLM_MODEL = os.getenv("VLLM_MODEL", "/mnt/d/models/Qwen2.5-7B-Instruct")

# vLLM just needs some non-empty API key
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

clients = [
    OpenAI(base_url=url.strip(), api_key=VLLM_API_KEY)
    for url in VLLM_URLS
    if url.strip()
]
clients_cycle = itertools.cycle(clients)


def get_client() -> OpenAI:
    return next(clients_cycle)


# ---------- FASTAPI APP ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev; tighten for real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.2  # low for medical-ish answers
    top_p: float = 0.9
    max_tokens: int = 512


class ChatResponse(BaseModel):
    reply: str


MEDICAL_SYSTEM_PROMPT = """
You are an AI medical information assistant. Your role is to:
- Provide general medical and health information.
- Help explain lab results, diseases, medications, and procedures in simple language.
- Encourage users to seek care from licensed healthcare professionals.
- Stay calm, respectful, and non-judgmental.

SAFETY RULES (VERY IMPORTANT):
- You are NOT a doctor and you are NOT a substitute for professional medical care.
- Do NOT give definitive diagnoses.
- Do NOT prescribe specific treatments, medications, dosages, or changes to existing prescriptions.
- If the user describes severe symptoms (e.g., chest pain, difficulty breathing, signs of stroke,
  suicidal thoughts, serious injury, high fever in infants), tell them to seek emergency medical care immediately.
- If you are unsure or the question is complex or high-risk, say that you cannot reliably answer
  and recommend they see a doctor.

STYLE:
- Be concise and structured (use bullet points when helpful).
- Use simple, patient-friendly language.
- At the end of each answer, include a short disclaimer like:
  "This is general information and not a substitute for professional medical advice. 
   Please consult a doctor for a diagnosis and personalized guidance."
""".strip()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    client = get_client()

    # Inject our medical system prompt at the beginning
    messages = [{"role": "system", "content": MEDICAL_SYSTEM_PROMPT}]
    # Append user/assistant history from frontend
    messages.extend([m.model_dump() for m in req.messages])

    completion = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=messages,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )
    reply = completion.choices[0].message.content
    return ChatResponse(reply=reply)


@app.get("/health")
async def health():
    return {"status": "ok", "vllm_backends": len(clients)}
