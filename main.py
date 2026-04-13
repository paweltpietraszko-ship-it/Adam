import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional, Literal

from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

import logging
from openai import OpenAI
from anthropic import Anthropic
from google import genai as google_genai

load_dotenv()

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("triangulum")

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = [
    "https://adam-production-89ef.up.railway.app",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.up\.railway\.app",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_api_key = os.getenv("GEMINI_API_KEY")

class Question(BaseModel):
    question: str
    mode: Literal["ogolny", "uczen"] = "ogolny"

    @field_validator('question')
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Pytanie nie moze byc puste.")
        if len(v) > 1000:
            raise ValueError("Pytanie jest zbyt dlugie (max 1000 znakow).")
        return v

class ChatMessage(BaseModel):
    question: str
    context: Optional[str] = ""
    verification: Optional[dict] = None
    history: Optional[list] = None
    mode: Optional[str] = "ogolny"

    @field_validator('question')
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Pytanie nie moze byc puste.")
        return v

def ask_openai(question: str) -> str:
    for attempt in range(2):
        try:
            r = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1500,
                messages=[{"role": "user", "content": question}]
            )
            result = r.choices[0].message.content
            return result if result and result.strip() else "[OPENAI ERROR] pusta odpowiedz"
        except Exception as e:
            logger.error(f"[OPENAI] Attempt {attempt+1} failed: {e}")
            if attempt == 1:
                return f"[OPENAI ERROR] {e}"
    return "[OPENAI ERROR] max retries"

def ask_claude(question: str) -> str:
    try:
        r = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,