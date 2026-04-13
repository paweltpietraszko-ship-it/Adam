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
            messages=[{"role": "user", "content": question}]
        )
        blocks = [b for b in r.content if hasattr(b, 'text') and b.text]
        return blocks[0].text if blocks else "[CLAUDE ERROR] brak tekstu w odpowiedzi"
    except Exception as e:
        logger.error(f"[CLAUDE] {e}", exc_info=True)
        return f"[CLAUDE ERROR] {e}"


gemini_client = google_genai.Client(api_key=gemini_api_key) if gemini_api_key else None

if not os.getenv("OPENAI_API_KEY"):
    logging.warning("[STARTUP] Brak OPENAI_API_KEY")
if not os.getenv("ANTHROPIC_API_KEY"):
    logging.warning("[STARTUP] Brak ANTHROPIC_API_KEY")
if not gemini_api_key:
    logging.warning("[STARTUP] Brak GEMINI_API_KEY — Gemini niedostepny")
else:
    logging.info(f"[STARTUP] Gemini client initialized, key: {gemini_api_key[:10]}...")


def ask_gemini(question: str) -> str:
    if not gemini_client:
        return "[GEMINI: brak klucza API]"
    try:
        logger.info(f"[GEMINI] Calling models/gemini-2.5-flash")
        response = gemini_client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=question
        )
        text = getattr(response, 'text', None)
        if text is None:
            try:
                text = response.candidates[0].content.parts[0].text
            except Exception:
                logger.error(f"[GEMINI] No text in response: {response}")
                return "[GEMINI ERROR] Brak tekstu w odpowiedzi"
        logger.info(f"[GEMINI] Success, length={len(text)}")
        return text
    except Exception as e:
        logger.error(f"[GEMINI] ERROR: {e}", exc_info=True)
        return f"[GEMINI ERROR] {e}"


def extract_verification(question: str, a: str, b: str, c: str) -> dict:
    gemini_section = (
        f"C (Gemini):\n{c}"
        if c and not any(c.startswith(p) for p in ["[GEMINI", "[CLAUDE", "[OPENAI"])
        else ""
    )

    prompt = f"""Zwroc TYLKO JSON. Zero prozy. Zero komentarzy. Tylko JSON.

Pytanie: {question}

A (Claude):
{a}

B (GPT):
{b}

{gemini_section}

Zasady:
WYSOKA = co najmniej 2 modele zgodne w kluczowych faktach, brak sprzecznosci
SREDNIA = 2 modele czesciowo zgodne LUB 1 sprzecznosc w szczegolach
NISKA = modele roznia sie w kluczowych twierdzeniach LUB bledy/timeouty

Schemat:
{{"certainty":"WYSOKA|SREDNIA|NISKA","certainty_reason":"jedno zdanie","facts_aligned":["fakt z co najmniej 2 odpowiedzi"],"contradictions":[{{"topic":"temat","positions":{{"claude":"stanowisko","gpt":"stanowisko","gemini":"stanowisko lub brak"}}}}],"uncertain":["teza spekulacyjna"],"models_count":2}}"""

    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
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
        
        working = 0
        for resp in [a, b, c]:
            if resp and isinstance(resp, str) and not any(
                resp.startswith(p) for p in ["[GEMINI", "[CLAUDE", "[OPENAI", "[SYNTHESIS"]
            ):
                working += 1
        result["models_count"] = working
        return result
    except json.JSONDecodeError:
        logger.error(f"[VERIFICATION] JSONDecodeError — raw: {raw[:200]}")
        working = sum(1 for r in [a,b,c] if r and not r.startswith("["))
        return {
            "certainty": "NISKA",
            "certainty_reason": "Blad parsowania JSON z weryfikatora",
            "facts_aligned": [], "contradictions": [], "uncertain": [],
            "models_count": working
        }
    except Exception as e:
        logger.error(f"[VERIFICATION] Error: {e}", exc_info=True)
        working = sum(1 for r in [a,b,c] if r and not r.startswith("["))
        return {
            "certainty": "NISKA",
            "certainty_reason": f"Blad ekstrakcji: {str(e)}",
            "facts_aligned": [], "contradictions": [], "uncertain": [],
            "models_count": working
        }


def render_synthesis(question: str, verification: dict, mode: str) -> str:
    if not verification:
        logger.warning("[SYNTHESIS] Pusta weryfikacja — uzywam wartosci domyslnych")
        verification = {
            "certainty": "NISKA",
            "certainty_reason": "Brak danych weryfikacyjnych",
            "facts_aligned": [], "contradictions": [], "uncertain": [],
            "models_count": 0
        }

    cert = verification.get("certainty", "NISKA")
    reason = verification.get("certainty_reason", "")
    facts = verification.get("facts_aligned", [])
    contras = verification.get("contradictions", [])
    uncertain = verification.get("uncertain", [])
    models_count = verification.get("models_count", 0)

    cert_labels = {
        "WYSOKA": "Modele sa zgodne — mozesz temu zaufac.",
        "SREDNIA": "Modele czesciowo sie roznia — sprawdz kluczowe fakty przed wazna decyzja.",
        "NISKA": "Modele sie roznia — potraktuj te odpowiedz jako punkt wyjscia, nie jako pewnik."
    }
    cert_label = cert_labels.get(cert, cert_labels["NISKA"])

    if mode == "uczen":
        prompt = f"""Masz wynik weryfikacji {models_count} modeli AI na pytanie: {question}

PEWNOSC: {cert} — {reason}
FAKTY ZGODNE: {facts}
SPRZECZNOSCI: {contras}
NIEPEWNE: {uncertain}
OCENA: {cert_label}

Napisz odpowiedz dla ucznia przygotowujacego sie do sprawdzianu.

Zasady:
1. Pisz pelnymi zdaniami. Zero list punktowanych.
2. Pierwsze zdanie: ocena zaufania i co uczen ma z tym zrobic.
3. Jezeli sa sprzecznosci: "Tu modele sie roznia — [opis]. Sprawdz w podreczniku."
4. Uzywaj TYLKO faktow z FAKTY ZGODNE.
5. Ton: madry starszy kolega tlumaczacy przed klasowka. Cieplo, konkretnie.

Struktura (akapity bez naglowkow):
1. Ocena zaufania i instrukcja dla ucznia.
2. Definicja egzaminacyjna z wyjasnieniem mechanizmu.
3. Przyklad ktory moze pojawic sie na sprawdzianie.
4. Typowy blad — jezeli sa sprzecznosci, opisz je jako pulapke.
5. Granice — czego nie trzeba wiedziec do tego sprawdzianu."""

    else:
        prompt = f"""Masz wynik weryfikacji {models_count} modeli AI na pytanie: {question}

PEWNOSC: {cert} — {reason}
FAKTY ZGODNE: {facts}
SPRZECZNOSCI: {contras}
NIEPEWNE: {uncertain}
OCENA: {cert_label}

Napisz synteze w trybie ogolnym.

Zasady:
1. Pisz pelnymi zdaniami. Zero list punktowanych.
2. Pierwsze zdanie: ocena zaufania.
3. Uzywaj TYLKO faktow z FAKTY ZGODNE.
4. Jezeli sa sprzecznosci, wspomnij o nich krotko.
5. Ton: rzeczowy, bez zargonu."""

    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"[SYNTHESIS] Error: {e}")
        return f"[SYNTHESIS ERROR] {e}"


def adam_reply(question: str, context: str, verification: dict, history: list) -> str:
    if not verification:
        verification = {"certainty": "nieznana", "facts_aligned": [], "contradictions": []}
    if not history:
        history = []

    cert = verification.get("certainty", "nieznana")
    facts = verification.get("facts_aligned", [])
    contras = verification.get("contradictions", [])

    system = f"""Jestes Adamem. Odpowiadasz na pytania dotyczace raportu ktory uzytkownik wlasnie otrzymal.

SYNTEZA RAPORTU:
{context}

DANE WERYFIKACYJNE:
Pewnosc: {cert}
Fakty zgodne miedzy modelami: {facts}
Sprzecznosci miedzy modelami: {contras}

Zasady bezwzgledne:
- Wolno ci uzywac TYLKO faktow z listy faktow zgodnych.
- Jezeli pytanie dotyczy sprzecznosci: "Tu modele sie roznia — [opisz roznice]."
- Jezeli pytanie wykracza poza raport: "Tego nie ma w zweryfikowanych zrodlach. Zadaj nowe pytanie do SILNIKA."
- Nie dodawaj wiedzy zewnetrznej. Nie spekuluj.
- Ton: naturalny, rzeczowy, cieplo. Pelne zdania z wyjasnieniem."""

    if not isinstance(history, list):
        history = []
    messages = [
        msg for msg in history[-20:]
        if isinstance(msg, dict) and "role" in msg and "content" in msg
    ]
    messages.append({"role": "user", "content": question})

    try:
        r = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            system=system,
            messages=messages
        )
        blocks = [b for b in r.content if hasattr(b, 'text') and b.text]
        return blocks[0].text if blocks else "[ADAM ERROR] brak tekstu"
    except Exception as e:
        logger.error(f"[ADAM] {e}", exc_info=True)
        return f"[ADAM ERROR] {e}"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "openai": openai_client is not None,
            "claude": claude_client is not None,
            "gemini": gemini_client is not None
        }
    }


@app.post("/ask")
@limiter.limit("30/minute")
def ask(q: Question, request: Request):
    try:
        start = time.time()
        with ThreadPoolExecutor(max_workers=3) as ex:
            fc = ex.submit(ask_claude, q.question)
            fo = ex.submit(ask_openai, q.question)
            fg = ex.submit(ask_gemini, q.question)

            claude_r = "[CLAUDE TIMEOUT]"
            openai_r = "[OPENAI TIMEOUT]"
            gemini_r = "[GEMINI TIMEOUT]"

            try: 
                claude_r = fc.result(timeout=30)
            except TimeoutError: 
                logger.warning("[MODEL] claude timeout")
            
            try: 
                openai_r = fo.result(timeout=30)
            except TimeoutError: 
                logger.warning("[MODEL] openai timeout")
            
            try: 
                gemini_r = fg.result(timeout=30)
            except TimeoutError: 
                logger.warning("[MODEL] gemini timeout")

        def is_error(resp):
            return not resp or any(resp.startswith(p) for p in ["[CLAUDE", "[OPENAI", "[GEMINI"]) or "TIMEOUT" in resp

        working_models = [r for r in [claude_r, openai_r, gemini_r] if not is_error(r)]
        logger.info(f"[ASK] working={len(working_models)}/3, claude_ok={not is_error(claude_r)}, openai_ok={not is_error(openai_r)}, gemini_ok={not is_error(gemini_r)}")
        
        if len(working_models) < 2:
            logger.warning(f"Za malo modeli: {len(working_models)}/3")
            return {
                "question": q.question, "mode": q.mode,
                "openai": openai_r, "claude": claude_r, "gemini": gemini_r,
                "synthesis": f"Za mało dostępnych modeli ({len(working_models)}/3). Wymagane co najmniej 2. Spróbuj ponownie.",
                "verification": {"certainty": "NISKA", "certainty_reason": f"Tylko {len(working_models)} model(e) odpowiedzia.", "facts_aligned": [], "contradictions": [], "uncertain": [], "models_count": len(working_models)},
                "status": "error"
            }

        verification_data = extract_verification(q.question, claude_r, openai_r, gemini_r)
        synthesis_text = render_synthesis(q.question, verification_data, q.mode)

        total_time = time.time() - start
        logger.info(f"[ASK] models={verification_data.get('models_count')}, certainty={verification_data.get('certainty')}, mode={q.mode}, total={total_time:.1f}s")
        return {
            "question": q.question,
            "mode": q.mode,
            "openai": openai_r,
            "claude": claude_r,
            "gemini": gemini_r,
            "synthesis": synthesis_text,
            "verification": verification_data,
            "status": "ok"
        }
    except Exception as e:
        logger.error(f"[ASK] Nieoczekiwany blad: {e}", exc_info=True)
        return {
            "question": q.question, "mode": q.mode,
            "openai": "", "claude": "", "gemini": "",
            "synthesis": "", "verification": {},
            "status": "error", "error": "Wewnętrzny błąd serwera."
        }


@app.post("/chat")
@limiter.limit("20/minute")
def chat(msg: ChatMessage, request: Request):
    try:
        answer = adam_reply(
            question=msg.question,
            context=msg.context or "",
            verification=msg.verification if msg.verification is not None else {},
            history=msg.history if msg.history is not None else []
        )
        return {"answer": answer, "status": "ok"}
    except Exception as e:
        logger.error(f"[CHAT] {e}", exc_info=True)
        return {"answer": "", "status": "error", "error": str(e)}


from fastapi.responses import FileResponse

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
