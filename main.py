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


gemini_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY")) if os.getenv("GEMINI_API_KEY") else None

if not os.getenv("OPENAI_API_KEY"):
    logging.warning("[STARTUP] Brak OPENAI_API_KEY")
if not os.getenv("ANTHROPIC_API_KEY"):
    logging.warning("[STARTUP] Brak ANTHROPIC_API_KEY")
if not gemini_api_key:
    logging.warning("[STARTUP] Brak GEMINI_API_KEY — Gemini niedostepny")


def ask_gemini(question: str, use_grounding: bool = False) -> str:
    if not gemini_client:
        return "[GEMINI: brak klucza API]"
    try:
        config = {}
        if use_grounding:
            from google.genai import types as genai_types
            config = {"tools": [genai_types.Tool(google_search=genai_types.GoogleSearch())]}

        response = gemini_client.models.generate_content(
            model="gemini-flash-latest",
            contents=question,
            **config
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
POTWIERDZONE ONLINE = Gemini (z dostepem do internetu) potwierdza fakty z co najmniej 1 modelu AI. Tylko dla pytan o twarde fakty.
WYSOKA = co najmniej 2 modele AI zgodne w kluczowych faktach, brak sprzecznosci
SREDNIA = 2 modele czesciowo zgodne LUB 1 sprzecznosc w szczegolach
NISKA = modele roznia sie w kluczowych twierdzeniach LUB bledy/timeouty

KLUCZOWA ZASADA: Jezeli model odpowiada ze nie zna aktualnych danych — jego odpowiedz POMIJASZ przy ocenie pewnosci. Uczciwy brak wiedzy nie jest sprzecznoscia.

Schemat:
{{"certainty":"POTWIERDZONE ONLINE|WYSOKA|SREDNIA|NISKA","certainty_reason":"jedno zdanie","facts_aligned":["fakt z co najmniej 2 modeli AI"],"contradictions":[{{"topic":"temat","positions":{{"claude":"stanowisko","gpt":"stanowisko","gemini":"stanowisko lub brak"}}}}],"uncertain":["teza spekulacyjna"],"models_count":2}}"""

    working = 0
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
def needs_web_search(question: str) -> bool:
    keywords = [
        "kurs", "cena", "dzisiaj", "teraz", "aktualny", "aktualna", "aktualne",
        "pogoda", "prognoza", "wynik meczu", "notowania", "walut", "akcji",
        "dzis", "dziś", "w tym tygodniu", "najnowsze", "ostatnie", "biezacy",
        "bieżący", "obecnie", "live", "jaki jest", "ile kosztuje"
    ]
    return any(kw in question.lower() for kw in keywords)


def classify_question(question: str) -> str:
    """Klasyfikuje pytanie: 'proste' lub 'zlozone'. Proste = jeden model wystarczy."""
    prompt = f"""Oceń czy to pytanie wymaga porównania wielu źródeł i weryfikacji, czy wystarczy jedna odpowiedź.

Pytanie: {question}

PROSTE = fakty encyklopedyczne, definicje, obliczenia, przepisy, instrukcje — rzeczy które mają jedną poprawną odpowiedź bez kontrowersji
ZLOZONE = tematy kontrowersyjne, polityczne, medyczne, naukowe z wieloma perspektywami, pytania gdzie modele AI mogą się mylić lub różnić

Odpowiedz TYLKO jednym słowem: PROSTE lub ZLOZONE"""
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        result = (r.choices[0].message.content or "").strip().upper()
        return "proste" if "PROSTE" in result else "zlozone"
    except Exception:
        return "zlozone"  # przy błędzie zawsze pełna triangulacja


def quick_synthesis(question: str, answer: str, mode: str) -> str:
    """Szybka synteza z jednego modelu — dla prostych pytań."""
    if mode == "uczen":
        prompt = f"""Pytanie: {question}

Odpowiedź: {answer}

Napisz krótkie wyjaśnienie dla ucznia. Bez nagłówków, bez list, 2-3 akapity prostym językiem.
Pierwsze zdanie: "To jest szybka odpowiedź z jednego źródła — bez pełnej weryfikacji."""
    else:
        prompt = f"""Pytanie: {question}

Odpowiedź: {answer}

Napisz zwięzłą odpowiedź dla dorosłego. Pierwsze zdanie musi brzmieć: "To jest szybka odpowiedź z jednego źródła — bez pełnej weryfikacji przez wiele modeli."
Użyj nagłówków: **SZYBKA ODPOWIEDŹ** i **GDZIE SĄ GRANICE**"""
    try:
        model = "claude-haiku-4-5" if mode == "uczen" else "claude-sonnet-4-6"
        r = claude_client.messages.create(
            model=model, max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        blocks = [b for b in r.content if hasattr(b, 'text') and b.text]
        return blocks[0].text if blocks else answer
    except Exception as e:
        return answer


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

Napisz odpowiedz dla ucznia lub mlodej osoby ktora chce szybko zrozumiec temat.

BEZWZGLEDNY ZAKAZ: nie uzywaj zadnych naglowkow markdown (**, ##, ###), zadnych list punktowanych ani numerowanych. Tylko ciagly tekst podzielony na akapity. Maksymalnie 4 krotkie akapity.

Zasady:
1. Jezyk prosty — tak jakbys tlumaczyl znajomemu przez telefon. Zadnych slow akademickich.
2. Pierwsze zdanie: czy mozna temu ufac i dlaczego w jednym zdaniu.
3. Najwazniejsza informacja w drugim zdaniu — konkretna, bez owijania w bawelne.
4. Jezeli sa sprzecznosci miedzy modelami: powiedz o tym wprost, krotko.
5. Ostatni akapit: co z tego wynika praktycznie dla tej osoby.
6. Ton: starszy brat lub siostra. Cieplo, bez pouczania, bez szkolnego jezyka.
7. Jezeli pytanie nie ma zwiazku ze szkola (np. kursy walut, aktualnosci) — nie wspominaj o sprawdzianie ani podreczniku.

KRYTYCZNA ZASADA dla sekcji gdzie opisujesz mechanizm:
Jezeli PEWNOSC = POTWIERDZONE ONLINE lub WYSOKA: opisz TYLKO jak to dziala. Zero watpliwosci w tym miejscu.
Jezeli PEWNOSC = SREDNIA lub NISKA: mozesz powiedziec ze nie wszyscy sie zgadzaja."""

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
[Kiedy ta wiedza nie dziala. Co zalezy od kontekstu.]

KRYTYCZNA ZASADA dla sekcji DLACZEGO TAK:
Jezeli PEWNOSC = POTWIERDZONE ONLINE lub WYSOKA: opisz TYLKO mechanizm lub kontekst faktu. Zero watpliwosci i zastrzezen w tej sekcji.
Jezeli PEWNOSC = SREDNIA lub NISKA: mozesz opisac roznice i niepewnosci."""

    try:
        model = "claude-haiku-4-5" if mode == "uczen" else "claude-sonnet-4-6"
        r = claude_client.messages.create(
            model=model,
            max_tokens=800,
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

Zasady:
- Jezeli pytanie dotyczy tresci raportu: odpowiedz opierajac sie na faktach zgodnych.
- Jezeli pytanie dotyczy sprzecznosci: "Tu modele sie roznia — [opisz roznice]."
- Jezeli pytanie wykracza POZA raport: odpowiedz na podstawie swojej wiedzy, ale ZAWSZE poprzedz odpowiedz zdaniem: "To wykracza poza zweryfikowany raport — odpowiadam jako Adam na podstawie wiedzy Claude, bez weryfikacji przez silnik. Jesli chcesz pewniejszej odpowiedzi, zadaj to pytanie bezposrednio do SILNIKA."
- Ton: naturalny, rzeczowy, cieplo. Pelne zdania z wyjasnieniem."""

    if not isinstance(history, list):
        history = []
    messages = [
        msg for msg in history[-20:]
        if isinstance(msg, dict)
        and msg.get("role") in ("user", "assistant")
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
            "openai": openai_client is not None,
            "claude": claude_client is not None,
            "gemini": gemini_client is not None,
            "grounding": gemini_client is not None
        }
    }


@app.post("/ask")
@limiter.limit("10/minute")
def ask(q: Question, request: Request):
    try:
        import time
        from datetime import datetime
        current_date = datetime.now().strftime("%A, %d %B %Y")
        question_with_date = f"[Dzisiaj jest: {current_date}]\n\n{q.question}"

        q_type = classify_question(q.question)
        use_grounding = needs_web_search(q.question)

        if q_type == "proste" and not use_grounding:
            # SZYBKA ŚCIEŻKA — tylko Claude
            t0 = time.time()
            claude_r = ask_claude(question_with_date)
            def is_error(resp):
                return not resp or any(resp.startswith(p) for p in ["[CLAUDE", "[OPENAI", "[GEMINI"]) or "TIMEOUT" in resp
            if is_error(claude_r):
                claude_r = ask_openai(question_with_date)
            synthesis = quick_synthesis(q.question, claude_r, q.mode)
            logger.info(f"[ASK-QUICK] total={time.time()-t0:.1f}s")
            return {
                "question": q.question, "mode": q.mode,
                "claude": claude_r, "openai": "", "gemini": "", "tavily": "",
                "synthesis": synthesis, "synthesis_uczen": "",
                "verification": {
                    "certainty": "SZYBKA",
                    "certainty_reason": "Proste pytanie — odpowiedź z jednego źródła bez pełnej weryfikacji.",
                    "facts_aligned": [], "contradictions": [], "uncertain": [],
                    "models_count": 1
                },
                "status": "ok", "quick": True
            }

        # PEŁNA TRIANGULACJA — Gemini z groundingiem gdy potrzeba internetu
        with ThreadPoolExecutor(max_workers=3) as ex:
            t0 = time.time()
            fc = ex.submit(ask_claude, question_with_date)
            fo = ex.submit(ask_openai, question_with_date)
            fg = ex.submit(ask_gemini, question_with_date, use_grounding)

            claude_r = "[CLAUDE TIMEOUT]"
            openai_r = "[OPENAI TIMEOUT]"
            gemini_r = "[GEMINI TIMEOUT]"

            try: claude_r = fc.result(timeout=30)
            except TimeoutError: pass

            try: openai_r = fo.result(timeout=30)
            except TimeoutError: pass

            try: gemini_r = fg.result(timeout=30)
            except TimeoutError: pass

            logger.info(f"[MODEL] total={time.time()-t0:.1f}s grounding={'yes' if use_grounding else 'no'}")

        def is_error(resp):
            return not resp or any(resp.startswith(p) for p in ["[CLAUDE", "[OPENAI", "[GEMINI"]) or "TIMEOUT" in resp

        working_models = [r for r in [claude_r, openai_r, gemini_r] if not is_error(r)]
        if len(working_models) < 1:
            return {
                "question": q.question, "mode": q.mode,
                "openai": openai_r, "claude": claude_r, "gemini": gemini_r,
                "synthesis": "Wszystkie modele niedostepne. Sprobuj ponownie za chwile.",
                "verification": {"certainty": "NISKA", "certainty_reason": "Brak odpowiedzi modeli.", "facts_aligned": [], "contradictions": [], "uncertain": [], "models_count": 0},
                "status": "error"
            }

        verification_data = extract_verification(q.question, claude_r, openai_r, gemini_r)

        synthesis = render_synthesis(q.question, verification_data, q.mode)

        logger.info(f"[ASK] models={verification_data.get('models_count')}, certainty={verification_data.get('certainty')}, mode={q.mode}")
        return {
            "question": q.question, "mode": q.mode,
            "openai": openai_r, "claude": claude_r, "gemini": gemini_r,
            "synthesis": synthesis, "synthesis_uczen": "",
            "verification": verification_data,
            "status": "ok", "quick": False
        }
    except Exception as e:
        logger.error(f"[ASK] Nieoczekiwany blad: {e}", exc_info=True)
        return {
            "question": q.question, "mode": q.mode,
            "openai": "", "claude": "", "gemini": "",
            "synthesis": "", "verification": {},
            "status": "error", "error": "Wewnetrzny blad serwera."
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


# =======================
# Static files
# =======================
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