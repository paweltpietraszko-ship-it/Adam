import os
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional, Literal
from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
import logging
from openai import OpenAI
from anthropic import Anthropic
from google import genai as google_genai
from tavily import TavilyClient
load_dotenv()
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

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
                model="gpt-4o-mini", max_tokens=1500,
                messages=[{"role": "user", "content": question}]
            )
            result = r.choices[0].message.content
            return result if result and result.strip() else "[OPENAI ERROR] pusta odpowiedz"
        except Exception as e:
            if attempt == 1:
                return f"[OPENAI ERROR] {e}"
    return "[OPENAI ERROR] max retries"

def ask_claude(question: str) -> str:
    try:
        r = claude_client.messages.create(
            model="claude-sonnet-4-6", max_tokens=2000,
            messages=[{"role": "user", "content": question}]
        )
        blocks = [b for b in r.content if hasattr(b, 'text') and b.text]
        return blocks[0].text if blocks else "[CLAUDE ERROR] brak tekstu w odpowiedzi"
    except Exception as e:
        return f"[CLAUDE ERROR] {e}"

gemini_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY")) if os.getenv("GEMINI_API_KEY") else None

if not os.getenv("OPENAI_API_KEY"):
    logging.warning("[STARTUP] Brak OPENAI_API_KEY")
if not os.getenv("ANTHROPIC_API_KEY"):
    logging.warning("[STARTUP] Brak ANTHROPIC_API_KEY")
if not gemini_api_key:
    logging.warning("[STARTUP] Brak GEMINI_API_KEY — Gemini niedostepny")
if not tavily_api_key:
    logging.warning("[STARTUP] Brak TAVILY_API_KEY — web search niedostepny")

def ask_gemini(question: str) -> str:
    if not gemini_client:
        return "[GEMINI: brak klucza API]"
    try:
        response = gemini_client.models.generate_content(
            model="gemini-flash-latest",
            contents=question
        )
        text = getattr(response, 'text', None)
        if text is None:
            try:
                text = response.candidates[0].content.parts[0].text
            except Exception as inner_e:
                logger.error(f"[GEMINI] Brak tekstu — error: {inner_e}")
                return "[GEMINI ERROR] Brak tekstu w odpowiedzi"
        return text
    except Exception as e:
        logger.error(f"[GEMINI] Pelny blad: {type(e).__name__}: {e}", exc_info=True)
        return f"[GEMINI ERROR] {e}"

def ask_tavily(question: str) -> str:
    if not tavily_client:
        return "[TAVILY: brak klucza API]"
    try:
        response = tavily_client.search(query=question, max_results=3)
        results = response.get("results", [])
        if not results:
            return "[TAVILY: brak wynikow]"
        parts = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            parts.append(f"[{title}] {content} (zrodlo: {url})")
        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"[TAVILY] Blad: {type(e).__name__}: {e}", exc_info=True)
        return f"[TAVILY ERROR] {e}"

def extract_verification(question: str, a: str, b: str, c: str, d: str = "") -> dict:
    gemini_section = (
        f"C (Gemini):\n{c}"
        if c and not any(c.startswith(p) for p in ["[GEMINI", "[CLAUDE", "[OPENAI"])
        else ""
    )
    tavily_section = (
        f"D (Internet — Tavily):\n{d}"
        if d and not any(d.startswith(p) for p in ["[TAVILY"])
        else ""
    )
    prompt = f"""Zwroc TYLKO JSON. Zero prozy. Zero komentarzy. Tylko JSON.
Pytanie: {question}
A (Claude):
{a}
B (GPT):
{b}
{gemini_section}
{tavily_section}
Zasady:
POTWIERDZONE ONLINE = Tavily (internet) potwierdza fakty z co najmniej 1 modelu AI
WYSOKA = co najmniej 2 modele AI zgodne w kluczowych faktach, brak sprzecznosci (bez Tavily)
SREDNIA = 2 modele czesciowo zgodne LUB 1 sprzecznosc w szczegolach
NISKA = modele roznia sie w kluczowych twierdzeniach LUB bledy/timeouty
Schemat:
{{"certainty":"POTWIERDZONE ONLINE|WYSOKA|SREDNIA|NISKA","certainty_reason":"jedno zdanie","facts_aligned":["fakt z co najmniej 2 odpowiedzi"],"contradictions":[{{"topic":"temat","positions":{{"claude":"stanowisko","gpt":"stanowisko","gemini":"stanowisko lub brak","tavily":"stanowisko lub brak"}}}}],"uncertain":["teza spekulacyjna"],"models_count":2}}"""

    working = 0
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini", temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = r.choices[0].message.content or ""
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        if not isinstance(result.get("facts_aligned"), list):
            result["facts_aligned"] = []
        if not isinstance(result.get("contradictions"), list):
            result["contradictions"] = []
        if not isinstance(result.get("uncertain"), list):
            result["uncertain"] = []
        for resp in [a, b, c, d]:
            if resp and isinstance(resp, str) and not any(
                resp.startswith(p) for p in ["[GEMINI", "[CLAUDE", "[OPENAI", "[TAVILY", "[SYNTHESIS"]
            ):
                working += 1
        result["models_count"] = working
        return result
    except json.JSONDecodeError:
        logger.error(f"[VERIFICATION] JSONDecodeError — raw: {raw[:200]}")
        return {"certainty": "NISKA", "certainty_reason": "Blad parsowania JSON", "facts_aligned": [], "contradictions": [], "uncertain": [], "models_count": working}
    except Exception as e:
        logger.error(f"[VERIFICATION] Error: {e}", exc_info=True)
        return {"certainty": "NISKA", "certainty_reason": f"Blad ekstrakcji: {str(e)}", "facts_aligned": [], "contradictions": [], "uncertain": [], "models_count": working}

def render_synthesis(question: str, verification: dict, mode: str) -> str:
    if not verification:
        logger.warning("[SYNTHESIS] Pusta weryfikacja")
        verification = {"certainty": "NISKA", "certainty_reason": "Brak danych", "facts_aligned": [], "contradictions": [], "uncertain": [], "models_count": 0}
    cert = verification.get("certainty", "NISKA")
    reason = verification.get("certainty_reason", "")
    facts = verification.get("facts_aligned", [])
    contras = verification.get("contradictions", [])
    uncertain = verification.get("uncertain", [])
    models_count = verification.get("models_count", 0)
    cert_labels = {
        "POTWIERDZONE ONLINE": "Odpowiedz potwierdzona przez aktualne zrodla internetowe.",
        "WYSOKA": "Modele sa zgodne — mozesz temu zaufac.",
        "SREDNIA": "Modele czesciowo sie roznia — sprawdz kluczowe fakty.",
        "NISKA": "Modele sie roznia — potraktuj jako punkt wyjscia, nie pewnik."
    }
    cert_label = cert_labels.get(cert, cert_labels["NISKA"])
    if mode == "uczen":
        prompt = f"""Masz wynik weryfikacji {models_count} modeli AI na pytanie: {question}
PEWNOSC: {cert} — {reason}
FAKTY ZGODNE: {facts}
SPRZECZNOSCI: {contras}
NIEPEWNE: {uncertain}
OCENA: {cert_label}
Napisz odpowiedz dla ucznia. BEZWZGLEDNY ZAKAZ naglowkow markdown i list. Tylko akapity.
5 akapitow: ocena zaufania, definicja, przyklad, typowy blad, granice tematu."""
    else:
        prompt = f"""Masz wynik weryfikacji {models_count} modeli AI na pytanie: {question}
PEWNOSC: {cert} — {reason}
FAKTY ZGODNE: {facts}
SPRZECZNOSCI: {contras}
NIEPEWNE: {uncertain}
OCENA: {cert_label}
Napisz odpowiedz dla doroslego. Pelne zdania, madry znajomy przy kawie.
Uzyj naglowkow: **SYNTEZA:** **CO Z TEGO WYNIKA** **DLACZEGO TAK** **TWARDE FAKTY** **CO WIEMY, A CZEGO NIE** **GDZIE SA GRANICE**"""
    try:
        r = claude_client.messages.create(
            model="claude-sonnet-4-6", max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        blocks = [b for b in r.content if hasattr(b, 'text') and b.text]
        return blocks[0].text if blocks else "[SYNTHESIS ERROR] brak tekstu"
    except Exception as e:
        return f"[SYNTHESIS ERROR] {e}"

def adam_reply(question: str, context: str, verification: dict, history: list) -> str:
    if not verification:
        verification = {"certainty": "nieznana", "facts_aligned": [], "contradictions": []}
    if not history:
        history = []
    cert = verification.get("certainty", "nieznana")
    facts = verification.get("facts_aligned", [])
    contras = verification.get("contradictions", [])
    system = f"""Jestes Adamem. Odpowiadasz na pytania dotyczace raportu.
SYNTEZA: {context}
Pewnosc: {cert}
Fakty zgodne: {facts}
Sprzecznosci: {contras}
Zasady: odpowiadaj na podstawie faktow zgodnych. Jezeli pytanie wykracza poza raport — poprzedz: "To wykracza poza zweryfikowany raport — odpowiadam jako Adam na podstawie wiedzy Claude, bez weryfikacji przez silnik."
Ton: naturalny, rzeczowy, cieplo."""
    if not isinstance(history, list):
        history = []
    messages = [msg for msg in history[-20:] if isinstance(msg, dict) and msg.get("role") in ("user", "assistant") and isinstance(msg.get("content"), str) and msg["content"].strip()]
    messages.append({"role": "user", "content": question})
    try:
        r = claude_client.messages.create(
            model="claude-haiku-4-5", max_tokens=800,
            system=system, messages=messages
        )
        blocks = [b for b in r.content if hasattr(b, 'text') and b.text]
        return blocks[0].text if blocks else "[ADAM ERROR] brak tekstu"
    except Exception as e:
        return f"[ADAM ERROR] {e}"

@app.get("/health")
def health():
    return {"status": "ok", "models": {"openai": openai_client is not None, "claude": claude_client is not None, "gemini": gemini_client is not None, "tavily": tavily_client is not None}}

@app.post("/ask")
@limiter.limit("10/minute")
def ask(q: Question, request: Request):
    try:
        import time
        with ThreadPoolExecutor(max_workers=4) as ex:
            t0 = time.time()
            fc = ex.submit(ask_claude, q.question)
            fo = ex.submit(ask_openai, q.question)
            fg = ex.submit(ask_gemini, q.question)
            ft = ex.submit(ask_tavily, q.question)
            claude_r = "[CLAUDE TIMEOUT]"
            openai_r = "[OPENAI TIMEOUT]"
            gemini_r = "[GEMINI TIMEOUT]"
            tavily_r = "[TAVILY TIMEOUT]"
            try: claude_r = fc.result(timeout=30)
            except TimeoutError: pass
            try: openai_r = fo.result(timeout=30)
            except TimeoutError: pass
            try: gemini_r = fg.result(timeout=30)
            except TimeoutError: pass
            try: tavily_r = ft.result(timeout=15)
            except TimeoutError: pass
            logger.info(f"[MODEL] total={time.time()-t0:.1f}s tavily={'ok' if not tavily_r.startswith('[') else 'err'}")
        def is_error(resp):
            return not resp or any(resp.startswith(p) for p in ["[CLAUDE", "[OPENAI", "[GEMINI", "[TAVILY"]) or "TIMEOUT" in resp
        working_models = [r for r in [claude_r, openai_r, gemini_r] if not is_error(r)]
        if len(working_models) < 2:
            logger.warning(f"Za malo modeli: {len(working_models)}/3")
            return {"question": q.question, "mode": q.mode, "openai": openai_r, "claude": claude_r, "gemini": gemini_r, "synthesis": f"Za malo modeli ({len(working_models)}/3).", "verification": {"certainty": "NISKA", "facts_aligned": [], "contradictions": [], "uncertain": [], "models_count": len(working_models)}, "status": "error"}
        verification_data = extract_verification(q.question, claude_r, openai_r, gemini_r, tavily_r)
        with ThreadPoolExecutor(max_workers=2) as ex2:
            f_ogolny = ex2.submit(render_synthesis, q.question, verification_data, "ogolny")
            f_uczen = ex2.submit(render_synthesis, q.question, verification_data, "uczen")
            synthesis_ogolny = f_ogolny.result(timeout=60)
            synthesis_uczen = f_uczen.result(timeout=60)
        logger.info(f"[ASK] models={verification_data.get('models_count')}, certainty={verification_data.get('certainty')}")
        return {"question": q.question, "mode": q.mode, "openai": openai_r, "claude": claude_r, "gemini": gemini_r, "tavily": tavily_r, "synthesis": synthesis_ogolny, "synthesis_uczen": synthesis_uczen, "verification": verification_data, "status": "ok"}
    except Exception as e:
        logger.error(f"[ASK] Blad: {e}", exc_info=True)
        return {"question": q.question, "mode": q.mode, "openai": "", "claude": "", "gemini": "", "synthesis": "", "verification": {}, "status": "error", "error": "Wewnetrzny blad serwera."}

@app.post("/chat")
@limiter.limit("20/minute")
def chat(msg: ChatMessage, request: Request):
    try:
        answer = adam_reply(question=msg.question, context=msg.context or "", verification=msg.verification if msg.verification is not None else {}, history=msg.history if msg.history is not None else [])
        return {"answer": answer, "status": "ok"}
    except Exception as e:
        return {"answer": "", "status": "error", "error": str(e)}

@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/index.html")
def index_html():
    return FileResponse("index.html")

@app.get("/manifest.json")
def manifest():
    return FileResponse("manifest.json", media_type="application/manifest+json")

@app.get("/service-worker.js")
def service_worker():
    return FileResponse("service-worker.js", media_type="application/javascript")

@app.get("/apple-touch-icon.png")
def apple_touch_icon():
    return FileResponse("apple-touch-icon.png", media_type="image/png")

@app.get("/icon-192.png")
def icon_192():
    return FileResponse("icon-192.png", media_type="image/png")

@app.get("/icon-512.png")
def icon_512():
    return FileResponse("icon-512.png", media_type="image/png")

@app.get("/icon-maskable.png")
def icon_maskable():
    return FileResponse("icon-maskable.png", media_type="image/png")

@app.get("/favicon.ico")
def favicon():
    return FileResponse("favicon.ico", media_type="image/x-icon")