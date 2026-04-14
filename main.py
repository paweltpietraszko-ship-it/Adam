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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("triangulum")

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Globalny licznik dzienny — ochrona przed nadużyciem
from collections import defaultdict
_daily_counter = {"date": "", "count": 0}
DAILY_LIMIT = 250

def check_daily_limit() -> bool:
    from datetime import date
    today = str(date.today())
    if _daily_counter["date"] != today:
        _daily_counter["date"] = today
        _daily_counter["count"] = 0
    _daily_counter["count"] += 1
    return _daily_counter["count"] <= DAILY_LIMIT

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


# =======================
# Schemas
# =======================
class Question(BaseModel):
    question: str
    mode: Literal["ogolny", "uczen"] = "ogolny"
    file_data: Optional[str] = None
    file_type: Optional[str] = None

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
def ask_openai(question: str, file_data: str = None, file_type: str = None) -> str:
    for attempt in range(2):
        try:
            content = []
            if file_data and file_type and file_type.startswith("image/"):
                content.append({"type": "image_url", "image_url": {"url": f"data:{file_type};base64,{file_data}"}})
            content.append({"type": "text", "text": question})
            r = openai_client.chat.completions.create(
                model="gpt-4o-mini", max_tokens=1500,
                messages=[{"role": "user", "content": content if len(content) > 1 else question}]
            )
            result = r.choices[0].message.content
            return result if result and result.strip() else "[OPENAI ERROR] pusta odpowiedz"
        except Exception as e:
            if attempt == 1:
                return f"[OPENAI ERROR] {e}"
    return "[OPENAI ERROR] max retries"


def ask_claude(question: str, file_data: str = None, file_type: str = None) -> str:
    try:
        content = []
        if file_data and file_type:
            if file_type == "application/pdf":
                content.append({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": file_data}})
            elif file_type.startswith("image/"):
                content.append({"type": "image", "source": {"type": "base64", "media_type": file_type, "data": file_data}})
        content.append({"type": "text", "text": question})
        r = claude_client.messages.create(
            model="claude-sonnet-4-6", max_tokens=2000,
            messages=[{"role": "user", "content": content}]
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


def ask_gemini(question: str, file_data: str = None, file_type: str = None) -> str:
    if not gemini_client:
        return "[GEMINI: brak klucza API]"
    try:
        if file_data and file_type:
            contents = [{"inline_data": {"mime_type": file_type, "data": file_data}}, question]
        else:
            contents = question
        response = gemini_client.models.generate_content(
            model="gemini-flash-latest",
            contents=contents
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


def needs_web_search(question: str) -> bool:
    """Tavily tylko gdy pytanie dotyczy aktualnych danych."""
    keywords = [
        "kurs", "cena", "dzisiaj", "teraz", "aktualny", "aktualna", "aktualne",
        "pogoda", "prognoza", "wynik meczu", "notowania", "walut", "akcji",
        "dzis", "dziś", "w tym tygodniu", "najnowsze", "ostatnie", "biezacy",
        "bieżący", "obecnie", "live", "jaki jest", "ile kosztuje"
    ]
    q = question.lower()
    return any(kw in q for kw in keywords)


import re
def strip_markdown(text: str) -> str:
    """Usuwa markdown z tekstu — druga warstwa bezpieczenstwa dla trybu uczen."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    return text.strip()


def fallback_synthesis(verification: dict, mode: str) -> str:
    """Awaryjna synteza gdy Claude niedostepny."""
    cert = verification.get("certainty", "NISKA")
    facts = verification.get("facts_aligned", [])
    if mode == "uczen":
        f = facts[0] if facts else "Brak zgodnych danych."
        return f"Pewność odpowiedzi: {cert}. Najważniejsze: {f} Szczegółowa synteza tymczasowo niedostępna."
    else:
        f_list = "\n- ".join(facts[:3]) if facts else "Brak zgodnych faktów."
        return f"**SYNTEZA:** Pewność: {cert}\n\n**TWARDE FAKTY**\n- {f_list}\n\n*(Synteza awaryjna — Claude tymczasowo niedostępny)*"


# =======================
# WARSTWA 1: Weryfikacja (JSON, niewidoczna)
# =======================
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
POTWIERDZONE ONLINE = Tavily (internet) potwierdza fakty z co najmniej 2 modeli AI jednoczesnie. Tylko dla pytan o twarde fakty (daty, nazwiska, wyniki, dane liczbowe). NIE dla opinii, analiz, porad.
WYSOKA = co najmniej 2 modele AI zgodne w kluczowych faktach, brak sprzecznosci
SREDNIA = 2 modele czesciowo zgodne LUB 1 sprzecznosc w szczegolach
NISKA = modele roznia sie w kluczowych twierdzeniach LUB bledy/timeouty

KLUCZOWA ZASADA: Jezeli model odpowiada ze nie zna aktualnych danych (kurs, cena, pogoda, wyniki na zywo) — jego odpowiedz POMIJASZ przy ocenie pewnosci. Uczciwy brak wiedzy nie jest sprzecznoscia.
Jezeli Tavily ma konkretne dane liczbowe a modele AI przyznaja brak wiedzy — uzyj danych Tavily jako podstawy i ocen pewnosc na POTWIERDZONE ONLINE jezeli dane sa spojne, SREDNIA jezeli rozne.

FILTR ZRODEL TAVILY: Jezeli Tavily zwraca sprzeczne dane z roznych zrodel — wybierz dane ze zrodel instytucjonalnych: domeny .gov .edu .org oficjalne kalendarze serwisy finansowe (nbp.pl bankier.pl investing.com). Odrzuc dane z social media (facebook.com twitter.com instagram.com tiktok.com) i blogów. Jezeli tylko social media dostepne — traktuj jako NISKA pewnosc.

Tavily jest sygnałem weryfikacyjnym — nie cytuj Tavily w facts_aligned chyba ze jest jedynym zrodlem konkretnej odpowiedzi (np. aktualna data, kurs waluty).
Jezeli Tavily jest niedostepny lub pusty: ignoruj go calkowicie.

Schemat:
{{"certainty":"POTWIERDZONE ONLINE|WYSOKA|SREDNIA|NISKA","certainty_reason":"jedno zdanie","facts_aligned":["fakt z co najmniej 2 modeli AI lub z Tavily gdy modele nie wiedza"],"contradictions":[{{"topic":"temat","positions":{{"claude":"stanowisko","gpt":"stanowisko","gemini":"stanowisko lub brak"}}}}],"uncertain":["teza spekulacyjna"],"models_count":2}}"""

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
def render_synthesis(question: str, verification: dict, mode: str, a: str = "", b: str = "", c: str = "") -> str:
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
        "POTWIERDZONE ONLINE": "Odpowiedz potwierdzona przez aktualne zrodla internetowe.",
        "WYSOKA": "Modele sa zgodne — mozesz temu zaufac.",
        "SREDNIA": "Modele czesciowo sie roznia — sprawdz kluczowe fakty przed wazna decyzja.",
        "NISKA": "Modele sie roznia — potraktuj te odpowiedz jako punkt wyjscia, nie jako pewnik."
    }
    cert_label = cert_labels.get(cert, cert_labels["NISKA"])

    # Przycinamy oryginalne odpowiedzi do 1000 znaków każda (ochrona przed limitem tokenów)
    a_short = (a[:1000] + "...") if len(a) > 1000 else a
    b_short = (b[:1000] + "...") if len(b) > 1000 else b
    c_short = (c[:1000] + "...") if len(c) > 1000 else c

    sources_section = ""
    if any([a_short, b_short, c_short]):
        parts = []
        if a_short and not a_short.startswith("[CLAUDE"): parts.append(f"Claude: {a_short}")
        if b_short and not b_short.startswith("[OPENAI"): parts.append(f"GPT: {b_short}")
        if c_short and not c_short.startswith("[GEMINI"): parts.append(f"Gemini: {c_short}")
        if parts:
            sources_section = "\n\nORYGINALNE ODPOWIEDZI MODELI (uzywaj jako kontekst, nie cytuj dosłownie):\n" + "\n\n".join(parts)

    if mode == "uczen":
        prompt = f"""Masz wynik weryfikacji {models_count} modeli AI na pytanie: {question}

PEWNOSC: {cert} — {reason}
FAKTY ZGODNE: {facts}
SPRZECZNOSCI: {contras}
NIEPEWNE: {uncertain}
OCENA: {cert_label}{sources_section}

WAZNE: Jezeli FAKTY ZGODNE zawieraja konkretna odpowiedz z internetu (aktualna data, kurs, wynik meczu, cena) — uzyj jej jako glownego faktu. Modele offline sa kontekstem, nie zrodlem. Nie pisz o ograniczeniach modeli jezeli fakty z internetu odpowiadaja na pytanie.

Napisz odpowiedz dla ucznia lub mlodej osoby ktora chce szybko zrozumiec temat.

BEZWZGLEDNY ZAKAZ: nie uzywaj zadnych naglowkow markdown (**, ##, ###), zadnych list punktowanych ani numerowanych. Tylko ciagly tekst podzielony na akapity. Maksymalnie 4 krotkie akapity.

Zasady:
1. Jezyk prosty — tak jakbys tlumaczyl znajomemu przez telefon. Zadnych slow akademickich.
2. Pierwsze zdanie: czy mozna temu ufac i dlaczego w jednym zdaniu.
3. Najwazniejsza informacja w drugim zdaniu — konkretna, bez owijania w bawelne.
4. Jezeli sa sprzecznosci miedzy modelami: powiedz o tym wprost, krotko.
5. Ostatni akapit: co z tego wynika praktycznie dla tej osoby.
6. Ton: starszy brat lub siostra. Cieplo, bez pouczania, bez szkolnego jezyka.
7. Jezeli pytanie nie ma zwiazku ze szkola — nie wspominaj o sprawdzianie ani podreczniku.

Jezeli PEWNOSC = POTWIERDZONE ONLINE lub WYSOKA: opisz TYLKO jak to dziala. Zero watpliwosci.
Jezeli PEWNOSC = SREDNIA lub NISKA: mozesz powiedziec ze nie wszyscy sie zgadzaja."""

    else:
        prompt = f"""Masz wynik weryfikacji {models_count} modeli AI na pytanie: {question}

PEWNOSC: {cert} — {reason}
FAKTY ZGODNE: {facts}
SPRZECZNOSCI: {contras}
NIEPEWNE: {uncertain}
OCENA: {cert_label}{sources_section}

WAZNE: Jezeli FAKTY ZGODNE zawieraja konkretna odpowiedz z internetu (aktualna data, kurs, wynik meczu, cena) — uzyj jej jako glownego faktu. Modele offline sa kontekstem, nie zrodlem. Nie pisz o ograniczeniach modeli jezeli fakty z internetu odpowiadaja na pytanie.

Napisz odpowiedz dla doroslego ktory chce zrozumiec temat.

Zasady:
1. Pisz pelnymi zdaniami z wyjasnieniem mechanizmu.
2. Pierwsze zdanie: ocena zaufania.
3. Sprzecznosci opisz jako roznice perspektyw, nie ukrywaj.
4. Liczby zawsze z kontekstem.
5. Ton: madry znajomy przy kawie. Rzeczowy, bezposredni, cieplo.

Uzyj tych naglowkow:

**SYNTEZA:** [1-2 zdania]

**CO Z TEGO WYNIKA**
[Konkretna odpowiedz]

**DLACZEGO TAK**
[Mechanizm przyczynowy — przy WYSOKA/POTWIERDZONE ONLINE: zero watpliwosci tutaj]

**TWARDE FAKTY**
[Liczby i przyklady. Jezeli brak: "Modele nie podaly zgodnych danych ilosciowych."]

**CO WIEMY, A CZEGO NIE**
- pewne: [tezy z co najmniej 2 modeli]
- czesciowe: [tezy z 1 modelu]
- niepewne: [sprzecznosci jako roznice perspektyw]

**GDZIE SA GRANICE**
[Kiedy ta wiedza nie dziala.]"""

    try:
        r = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        blocks = [b for b in r.content if hasattr(b, 'text') and b.text]
        result = blocks[0].text if blocks else None
        if not result:
            return fallback_synthesis(verification, mode)
        if mode == "uczen":
            result = strip_markdown(result)
        return result
    except Exception as e:
        logger.error(f"[SYNTHESIS] Claude error: {e}", exc_info=True)
        return fallback_synthesis(verification, mode)


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
            "tavily": tavily_client is not None
        }
    }


@app.post("/ask")
@limiter.limit("10/minute")
def ask(q: Question, request: Request):
    if not check_daily_limit():
        return {"question": q.question, "status": "error", "error": "Dzienny limit zapytań wyczerpany. Spróbuj jutro."}
    try:
        import time
        from datetime import datetime
        # Wstrzyknij aktualną datę — modele nie muszą zgadywać
        current_date = datetime.now().strftime("%A, %d %B %Y")
        question_with_date = f"[Dzisiaj jest: {current_date}]\n\n{q.question}"

        use_tavily = needs_web_search(q.question) and not q.file_data
        tavily_r = "[TAVILY: pominiety]"

        with ThreadPoolExecutor(max_workers=4) as ex:
            t0 = time.time()
            fc = ex.submit(ask_claude, question_with_date, q.file_data, q.file_type)
            fo = ex.submit(ask_openai, question_with_date, q.file_data, q.file_type)
            fg = ex.submit(ask_gemini, question_with_date, q.file_data, q.file_type)
            ft = ex.submit(ask_tavily, q.question) if use_tavily else None

            claude_r = "[CLAUDE TIMEOUT]"
            openai_r = "[OPENAI TIMEOUT]"
            gemini_r = "[GEMINI TIMEOUT]"

            try: claude_r = fc.result(timeout=50 if q.file_data else 30)
            except TimeoutError: pass

            try: openai_r = fo.result(timeout=30)
            except TimeoutError: pass

            try: gemini_r = fg.result(timeout=30)
            except TimeoutError: pass

            if ft is not None:
                try: tavily_r = ft.result(timeout=15)
                except TimeoutError: pass

            logger.info(f"[MODEL] total={time.time()-t0:.1f}s file={'yes' if q.file_data else 'no'} tavily={'yes' if use_tavily else 'skip'}")

        def is_error(resp):
            return not resp or any(resp.startswith(p) for p in ["[CLAUDE", "[OPENAI", "[GEMINI", "[TAVILY"]) or "TIMEOUT" in resp

        working_models = [r for r in [claude_r, openai_r, gemini_r] if not is_error(r)]
        if len(working_models) < 2:
            logger.warning(f"Za malo modeli: {len(working_models)}/3")
            return {
                "question": q.question, "mode": q.mode,
                "openai": openai_r, "claude": claude_r, "gemini": gemini_r,
                "synthesis": f"Za malo dostepnych modeli ({len(working_models)}/3). Wymagane co najmniej 2. Sprobuj ponownie.",
                "verification": {"certainty": "NISKA", "certainty_reason": f"Tylko {len(working_models)} model(e) odpowiedzia.", "facts_aligned": [], "contradictions": [], "uncertain": [], "models_count": len(working_models)},
                "status": "error"
            }

        verification_data = extract_verification(q.question, claude_r, openai_r, gemini_r, tavily_r)
        
        # Generuj obie syntezy rownolegnie
        with ThreadPoolExecutor(max_workers=2) as ex2:
            f_ogolny = ex2.submit(render_synthesis, q.question, verification_data, "ogolny", claude_r, openai_r, gemini_r)
            f_uczen = ex2.submit(render_synthesis, q.question, verification_data, "uczen", claude_r, openai_r, gemini_r)
            synthesis_ogolny = f_ogolny.result(timeout=60)
            synthesis_uczen = f_uczen.result(timeout=60)

        logger.info(f"[ASK] models={verification_data.get(chr(39)+'models_count'+chr(39))}, certainty={verification_data.get(chr(39)+'certainty'+chr(39))}, mode=dual")
        return {
            "question": q.question,
            "mode": q.mode,
            "openai": openai_r,
            "claude": claude_r,
            "gemini": gemini_r,
            "tavily": tavily_r,
            "synthesis": synthesis_ogolny,
            "synthesis_uczen": synthesis_uczen,
            "verification": verification_data,
            "status": "ok"
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