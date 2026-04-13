import os
import json
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
    # "null" usuniete — niebezpieczne w produkcji
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


# =======================
# Schemas
# =======================
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


# =======================
# Model calls
# =======================
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
        return f"[CLAUDE ERROR] {e}"


# Klient Gemini inicjalizowany raz na poziomie modulu
gemini_client = google_genai.Client(api_key=gemini_api_key) if gemini_api_key else None

# Loguj stan kluczy przy starcie
if not os.getenv("OPENAI_API_KEY"):
    logging.warning("[STARTUP] Brak OPENAI_API_KEY")
if not os.getenv("ANTHROPIC_API_KEY"):
    logging.warning("[STARTUP] Brak ANTHROPIC_API_KEY")
if not gemini_api_key:
    logging.warning("[STARTUP] Brak GEMINI_API_KEY — Gemini niedostepny")


def ask_gemini(question: str) -> str:
    if not gemini_client:
        return "[GEMINI: brak klucza API]"
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[{"parts": [{"text": question}]}]
        )
        text = getattr(response, 'text', None)
        if text is None:
            try:
                text = response.candidates[0].content.parts[0].text
            except Exception:
                return "[GEMINI ERROR] Brak tekstu w odpowiedzi"
        return text
    except Exception as e:
        return f"[GEMINI ERROR] {e}"


# =======================
# WARSTWA 1: Weryfikacja (JSON, niewidoczna)
# =======================
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
            temperature=0,  # deterministic JSON output
            messages=[{"role": "user", "content": prompt}]
        )
        raw = r.choices[0].message.content or ""
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        # Validate expected structure
        if not isinstance(result.get("facts_aligned"), list):
            result["facts_aligned"] = []
        if not isinstance(result.get("contradictions"), list):
            result["contradictions"] = []
        if not isinstance(result.get("uncertain"), list):
            result["uncertain"] = []
        # Count only models that actually responded without error
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
        return {
            "certainty": "NISKA",
            "certainty_reason": "Blad parsowania JSON z weryfikatora",
            "facts_aligned": [], "contradictions": [], "uncertain": [],
            "models_count": working
        }
    except Exception as e:
        logger.error(f"[VERIFICATION] Error: {e}", exc_info=True)
        return {
            "certainty": "NISKA",
            "certainty_reason": f"Blad ekstrakcji: {str(e)}",
            "facts_aligned": [], "contradictions": [], "uncertain": [],
            "models_count": working
        }


# =======================
# WARSTWA 2: Prezentacja (proza, widoczna)
# =======================
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

Napisz odpowiedz dla doroslego ktory chce zrozumiec temat.

Zasady:
1. Pisz pelnymi zdaniami z wyjasnieniem mechanizmu.
2. Pierwsze zdanie: ocena zaufania.
3. Sprzecznosci opisz jako roznice perspektyw, nie ukrywaj.
4. Liczby zawsze z kontekstem.
5. Uzywaj TYLKO faktow z FAKTY ZGODNE.
6. Ton: madry znajomy przy kawie. Rzeczowy, bezposredni, cieplo.
7. Badz szczodry w wyjasnieniach.

Uzyj tych naglowkow:

**SYNTEZA:** [1-2 zdania]

**CO Z TEGO WYNIKA**
[Konkretna odpowiedz z faktow zgodnych]

**DLACZEGO TAK**
[Mechanizm przyczynowy]

**TWARDE FAKTY**
[Liczby i przyklady z kontekstem. Jezeli brak: "Modele nie podaly zgodnych danych ilosciowych."]

**CO WIEMY, A CZEGO NIE**
- pewne: [tezy z co najmniej 2 modeli]
- czesciowe: [tezy z 1 modelu]
- niepewne: [sprzecznosci jako roznice perspektyw]

**GDZIE SA GRANICE**
[Kiedy ta wiedza nie dziala. Co zalezy od kontekstu.]"""

    try:
        r = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        blocks = [b for b in r.content if hasattr(b, 'text') and b.text]
        return blocks[0].text if blocks else "[SYNTHESIS ERROR] brak tekstu"
    except Exception as e:
        return f"[SYNTHESIS ERROR] {e}"


# =======================
# Adam
# =======================
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
        msg for msg in history[-20:]  # max 20 wiadomosci
        if isinstance(msg, dict)
        and msg.get("role") in ("user", "assistant")  # blokuj prompt injection przez role=system
        and isinstance(msg.get("content"), str)
        and msg["content"].strip()
    ]
    messages.append({"role": "user", "content": question})

    try:
        r = claude_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=800,
            system=system,
            messages=messages
        )
        blocks = [b for b in r.content if hasattr(b, 'text') and b.text]
        return blocks[0].text if blocks else "[ADAM ERROR] brak tekstu"
    except Exception as e:
        return f"[ADAM ERROR] {e}"


# =======================
# API
# =======================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            # Sprawdza konfiguracje, nie polaczenie z API
            "openai": openai_client is not None,
            "claude": claude_client is not None,
            "gemini": gemini_client is not None
        }
    }


@app.post("/ask")
@limiter.limit("10/minute")
def ask(q: Question, request: Request):
    try:
        import time
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=3) as ex:
            t_claude = time.time()
            fc = ex.submit(ask_claude, q.question)
            t_openai = time.time()
            fo = ex.submit(ask_openai, q.question)
            t_gemini = time.time()
            fg = ex.submit(ask_gemini, q.question)

            claude_r = "[CLAUDE TIMEOUT]"
            openai_r = "[OPENAI TIMEOUT]"
            gemini_r = "[GEMINI TIMEOUT]"

            try: claude_r = fc.result(timeout=12)
            except TimeoutError: pass
            logger.info(f"[MODEL] claude={time.time()-t_claude:.1f}s")

            try: openai_r = fo.result(timeout=10)
            except TimeoutError: pass
            logger.info(f"[MODEL] openai={time.time()-t_openai:.1f}s")

            try: gemini_r = fg.result(timeout=10)
            except TimeoutError: pass
            logger.info(f"[MODEL] gemini={time.time()-t_gemini:.1f}s")

        # Sprawdz czy wszystkie modele sa niedostepne
        def is_error(resp):
            return not resp or any(resp.startswith(p) for p in ["[CLAUDE", "[OPENAI", "[GEMINI"]) or "TIMEOUT" in resp

        working_models = [r for r in [claude_r, openai_r, gemini_r] if not is_error(r)]
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

        logger.info(f"[ASK] models={verification_data.get('models_count')}, certainty={verification_data.get('certainty')}, mode={q.mode}")
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
        return {"answer": "", "status": "error", "error": str(e)}


@app.get("/")
def root():
    html = """<!DOCTYPE html>
<html lang="pl"><head><meta charset="UTF-8"><title>Triangulum</title>
<style>body{background:#0a0c0f;color:#c8cdd6;font-family:monospace;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}.box{text-align:center;padding:24px}.title{font-size:24px;color:#5b9bd5;margin-bottom:8px}.sub{font-size:11px;color:#2d4a6e;letter-spacing:.15em;text-transform:uppercase;margin-bottom:24px}.ep{background:#0d1015;border:1px solid #1a1f26;border-radius:3px;padding:10px 16px;margin-bottom:8px;font-size:12px;color:#3a6a8a}.status{font-size:10px;color:#3a7a5a;margin-top:16px}</style>
</head><body><div class="box">
<div class="title">TRIANGULUM</div>
<div class="sub">System weryfikacji multi-model</div>
<div class="ep">POST /ask</div><div class="ep">POST /chat</div><div class="ep">GET /health</div>
<div class="status">Backend aktywny · Claude · GPT-4o-mini · Gemini</div>
</div></body></html>"""
    return Response(content=html, media_type="text/html")