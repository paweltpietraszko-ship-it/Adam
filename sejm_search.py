# v2.7 — extract_legal_keywords: frazy wielowyrazowe, art. N, semantic_terms ≥4 znaki
"""
Triangulum — sejm_search.py  v1
================================
Wyszukiwanie aktualnych przepisów przez oficjalne API Sejmu RP.
Źródło: https://api.sejm.gov.pl (Kancelaria Sejmu, dane publiczne)

Cel: dostarczyć modelom aktualny tekst przepisu jako kontekst,
zamiast odpowiadania z wiedzy treningowej z cutoffem.

Zasada RODO:
  Do zewnętrznego serwera (api.sejm.gov.pl) wychodzą TYLKO słowa kluczowe
  wyekstrahowane z pytania — nigdy treść dokumentu klienta, nigdy pseudonimy.
  api.sejm.gov.pl to serwer publiczny Sejmu RP, nie podmiot przetwarzający
  dane osobowe.

Pipeline integracji z main.py:
  1. Wykryj że pytanie dotyczy aktualnego stanu prawa (sygnały: "obowiązujący",
     "aktualny", "od kiedy", "zmiana", "nowelizacja", rok bieżący itp.)
  2. Wyekstrahuj słowa kluczowe (tytuł ustawy, numer artykułu, temat)
  3. Wywołaj search_acts() → lista aktów
  4. Pobierz tekst najlepszego trafieniea przez get_act_text()
  5. Przekaż jako kontekst do modeli w prompcie systemowym

Ograniczenia:
  - Wyszukiwanie po tytule, nie po treści artykułów — dla pytań
    "co mówi art. X ustawy Y" działa dobrze, dla "jaki przepis reguluje X"
    wymaga dobrego doboru słów kluczowych
  - Teksty HTML wymagają stripowania tagów — jakość zależy od struktury
    danego aktu
  - API nie ma full-text search po treści artykułów (tylko tytuły + słowa kluczowe)
  - Interpretacje KIS i orzeczenia NSA: API Sejmu ich nie zawiera
"""

import re
import ssl
import warnings
import logging
import urllib.request
import urllib.parse
import urllib.error
import json
import html
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# TODO: zastąpić właściwym certyfikatem CA przed wdrożeniem produkcyjnym
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

logger = logging.getLogger("triangulum.sejm_search")

# ============================================================
# Konfiguracja
# ============================================================

BASE_URL      = "https://api.sejm.gov.pl/eli"
TIMEOUT_S     = 10          # timeout HTTP per request
MAX_TEXT_CHARS = 8_000      # limit kontekstu na akt — nie zalewamy okna modelu
MAX_RESULTS   = 5           # domyślna liczba wyników wyszukiwania

# Sygnały w pytaniu że potrzebna jest aktualna wiedza prawna
# [FIX] Zmiana strategii: inkluzja zamiast wykluczeń.
# Poprzednia NON_LEGAL_SIGNALS była za krótka — "Ile wynosi 2+2?" zwracało True.
# Sejm API wywoływane TYLKO gdy pytanie zawiera znane terminy prawne.
LEGAL_SIGNALS = [
    # Z polskimi znakami
    "ustawa", "kodeks", "rozporządzenie", "dyrektywa", "przepis", "przepisy",
    "art.", "artykuł", "paragraf", "§",
    "k.p.", "k.c.", "ksh", "kpa",
    "vat", "pit", "cit", "zus", "składki", "składka", "podatek", "podatki",
    "ubezpieczenie", "emerytura", "renta",
    "pracownik", "pracodawca", "wynagrodzenie", "urlop", "zwolnienie",
    "umowa o pracę", "umowa zlecenia", "umowa o dzieło",
    "zleceniobiorca", "zleceniodawca", "wypowiedzenie", "etat", "kodeks pracy",
    "najem", "dzierżawa", "najemca", "wynajmujący", "sprzedaż", "darowizna",
    "testament", "klauzula", "odszkodowanie", "umowa najmu",
    "spółka", "działalność gospodarcza", "regon", "nip", "krs",
    "obowiązki", "uprawnienia", "odpowiedzialność", "roszczenie",
    "poufność", "zakaz konkurencji", "karencja", "pełnomocnictwo",
    "prawo pracy", "kodeks cywilny",
    # Bez polskich znaków — użytkownicy często piszą bez diakrytyków
    "rozporzadzenie", "przepisy", "artykul",
    "skladki", "skladka", "podatek",
    "pracownik", "pracodawca", "wynagrodzenie",
    "umowa o prace", "umowa zlecenia", "umowa o dzielo",
    "zleceniobiorca", "zleceniodawca", "wypowiedzenie",
    "dzierzawa", "najemca", "wynajmujacy", "sprzedaz",
    "spolka", "dzialalnosc gospodarcza",
    "obowiazki", "uprawnienia", "odpowiedzialnosc", "roszczenie",
    "poufnosc", "zakaz konkurencji", "pelnomocnictwo",
    "prawo pracy", "kodeks cywilny",
]


# ============================================================
# Dataclass wyników
# ============================================================

@dataclass
class ActResult:
    title:           str
    display_address: str          # np. "Dz.U. 2023 poz. 1234"
    status:          str          # "obowiązujący" / "uchylony" / ...
    in_force:        str          # "IN_FORCE" / "NOT_IN_FORCE" / "UNKNOWN"
    eli:             str          # np. "DU/2023/1234"
    publisher:       str          # "DU" lub "MP"
    year:            int
    position:        int
    entry_into_force: Optional[str] = None
    keywords:        list[str]    = field(default_factory=list)
    text_html:       bool         = False
    url_isap:        str          = ""

    @property
    def is_active(self) -> bool:
        return self.in_force == "IN_FORCE"

    @property
    def isap_url(self) -> str:
        # Format: WDU + rok(4) + tom(4, po 2012 zawsze 0000) + pozycja(4)
        return (
            f"https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp"
            f"?id=WDU{self.year:04d}0000{self.position:04d}"
        )


@dataclass
class SearchResult:
    query:      str
    total:      int
    acts:       list[ActResult]   = field(default_factory=list)
    error:      Optional[str]     = None


# ============================================================
# HTTP helper — bez zewnętrznych zależności (tylko stdlib)
# ============================================================

def _get_json(url: str) -> dict | list:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "Triangulum/1.0"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_html(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"Accept": "text/html", "User-Agent": "Triangulum/1.0"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S, context=_SSL_CTX) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _strip_html(raw: str) -> str:
    """
    Usuwa tagi HTML, dekoduje encje, normalizuje whitespace.
    Nie używa zewnętrznych parserów — działa na surowym stringu.
    """
    # Usuń skrypty i style
    raw = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", raw,
                 flags=re.DOTALL | re.IGNORECASE)
    # Zamień blokowe tagi na newline
    raw = re.sub(r"<(br|p|div|h\d|li|tr)[^>]*>", "\n", raw, flags=re.IGNORECASE)
    # Usuń pozostałe tagi
    raw = re.sub(r"<[^>]+>", "", raw)
    # Dekoduj encje HTML (&amp; &nbsp; itp.)
    raw = html.unescape(raw)
    # Normalizuj whitespace
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r" {2,}", " ", raw)
    return raw.strip()


# ============================================================
# Wyszukiwanie aktów
# ============================================================

def _fetch_acts(params: dict, label: str) -> list:
    """
    Pomocnicza — wykonuje jedno zapytanie do /acts/search i zwraca listę items.
    Zwraca [] przy błędzie (nie rzuca wyjątku — awaria jednej gałęzi nie blokuje drugiej).
    """
    url = f"{BASE_URL}/acts/search?" + urllib.parse.urlencode(params)
    logger.info(f"sejm_search: GET [{label}] {url}")
    try:
        data = _get_json(url)
        return data.get("items", [])
    except Exception as e:
        logger.warning(f"sejm_search: [{label}] błąd — {e}")
        return []


def _items_to_acts(items: list) -> list[ActResult]:
    """Konwertuje surowe items z API na listę ActResult."""
    acts = []
    for item in items:
        act = ActResult(
            title=            item.get("title", ""),
            display_address=  item.get("displayAddress", ""),
            status=           item.get("status", ""),
            in_force=         item.get("inForce", "UNKNOWN"),
            eli=              item.get("ELI", ""),
            publisher=        item.get("publisher", "DU"),
            year=             item.get("year", 0),
            position=         item.get("pos", 0),
            entry_into_force= item.get("entryIntoForce"),
            keywords=         item.get("keywords", []),
            text_html=        item.get("textHTML", False),
        )
        acts.append(act)
        logger.debug(
            f"sejm_search: [{act.in_force}] {act.display_address} — {act.title[:60]}"
        )
    return acts


def search_acts(
    query: str,
    in_force_only: bool = True,
    max_results: int = MAX_RESULTS,
    publisher: str = "DU",
) -> SearchResult:
    """
    Szuka aktów prawnych równolegle po tytule (title=) i hasłach tematycznych (keyword=).

    API Sejmu nie robi full-text search — samo title= często nie trafia w akt
    bo formalne tytuły ustaw różnią się od słów kluczowych pytania.
    Przykład: "minimalne wynagrodzenie" nie trafi w
    "Ustawa z dnia 10 października 2002 r. o minimalnym wynagrodzeniu za pracę"
    ale trafi w keyword="wynagrodzenia minimalne" z taksonomii Sejmu.

    Strategia:
      - Dwa równoległe zapytania: title= i keyword=
      - Wyniki deduplikowane po ELI (unikalny identyfikator aktu)
      - Akty z keyword= priorytetyzowane (taksonomia Sejmu jest semantycznie bliższa)
      - Przy błędzie obu zapytań zwraca SearchResult z error != None
    """
    base_params = {
        "publisher": publisher,
        "limit":     str(max_results),
        "sortBy":    "change",
        "sortDir":   "desc",
    }
    if in_force_only:
        base_params["inForce"] = "1"

    # Zapytanie 1: po tytule (stara ścieżka — działa dla "kodeks pracy" itp.)
    title_params = {**base_params, "title": query}
    # Zapytanie 2: po hasłach tematycznych (nowa ścieżka — semantyczna taksonomia Sejmu)
    keyword_params = {**base_params, "keyword": query}

    items_title   = _fetch_acts(title_params,   "title")
    items_keyword = _fetch_acts(keyword_params, "keyword")

    if not items_title and not items_keyword:
        msg = f"Brak wyników dla zapytania '{query}' (title= i keyword=)"
        logger.warning(f"sejm_search: {msg}")
        return SearchResult(query=query, total=0, error=msg)

    # Deduplikacja po ELI — keyword= pierwsze (wyższy priorytet semantyczny)
    seen_eli: set[str] = set()
    merged_items = []
    for item in (items_keyword + items_title):
        eli = item.get("ELI", "")
        if eli and eli in seen_eli:
            continue
        if eli:
            seen_eli.add(eli)
        merged_items.append(item)

    merged_items = merged_items[:max_results]
    acts = _items_to_acts(merged_items)

    logger.info(
        f"sejm_search: title={len(items_title)} keyword={len(items_keyword)} "
        f"po deduplikacji={len(acts)}"
    )
    return SearchResult(query=query, total=len(acts), acts=acts)


# ============================================================
# Pobieranie tekstu aktu
# ============================================================

def get_act_text(
    publisher: str,
    year: int,
    position: int,
    max_chars: int = MAX_TEXT_CHARS,
) -> Optional[str]:
    """
    Pobiera tekst aktu w formacie HTML i zwraca jako plain text.

    Zwraca None jeśli akt nie ma tekstu HTML lub wystąpił błąd.
    Tekst jest obcięty do max_chars — nie zalewamy okna kontekstu modelu.
    """
    url = f"{BASE_URL}/acts/{publisher}/{year}/{position}/text.html"
    logger.info(f"sejm_search: pobieranie tekstu — {url}")

    try:
        raw = _get_html(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.warning(
                f"sejm_search: brak tekstu HTML dla {publisher}/{year}/{position}"
            )
        else:
            logger.error(f"sejm_search: HTTP {e.code} przy pobieraniu tekstu")
        return None
    except Exception as e:
        logger.error(f"sejm_search: błąd pobierania tekstu: {e}")
        return None

    text = _strip_html(raw)

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[...tekst skrócony do {max_chars} znaków]"

    return text


def get_act_fragment(
    publisher: str,
    year: int,
    position: int,
    tree_path: str,
) -> Optional[str]:
    """
    Pobiera konkretny fragment aktu (artykuł, ustęp, punkt).

    tree_path — ścieżka w strukturze dokumentu, np.:
      "art=22"                        → art. 22
      "rozdzial=3/art=15/ustep=2"     → rozdz. 3, art. 15 ust. 2
      "art=4/para=1/ustep=3/punkt=2"  → art. 4 §1 ust. 3 pkt 2

    Elementy ścieżki: ksiega, tytul, dzial, rozdzial, oddzial,
                      artykul (lub art), ustep, paragraf, punkt, litera
    """
    url = (
        f"{BASE_URL}/acts/{publisher}/{year}/{position}"
        f"/text.html/{tree_path}"
    )
    logger.info(f"sejm_search: fragment — {url}")

    try:
        raw = _get_html(url)
        return _strip_html(raw)
    except urllib.error.HTTPError as e:
        logger.warning(
            f"sejm_search: HTTP {e.code} dla fragmentu {tree_path}"
        )
        return None
    except Exception as e:
        logger.error(f"sejm_search: błąd pobierania fragmentu: {e}")
        return None


def get_act_text_pdf(
    publisher: str,
    year: int,
    position: int,
    max_chars: int = MAX_TEXT_CHARS,
) -> Optional[str]:
    """
    Pobiera tekst aktu z PDF gdy textHTML=False (np. rozporządzenia RM).

    Używa pdfplumber jeśli dostępny, fallback do pypdf, fallback do None.
    Rozporządzenia RM ustalające stawki mają zazwyczaj 3-5 paragrafów —
    cały tekst jest krótki i nie wymaga keyword scoring.

    Zwraca None jeśli PDF niedostępny lub parsowanie nie powiodło się.
    """
    url = f"{BASE_URL}/acts/{publisher}/{year}/{position}/text.pdf"
    logger.info(f"sejm_search: pobieranie PDF — {url}")

    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/pdf", "User-Agent": "Triangulum/1.0"}
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_S, context=_SSL_CTX) as resp:
            pdf_bytes = resp.read()
    except urllib.error.HTTPError as e:
        logger.warning(f"sejm_search: HTTP {e.code} przy pobieraniu PDF")
        return None
    except Exception as e:
        logger.warning(f"sejm_search: błąd pobierania PDF: {e}")
        return None

    # Próba 1: pdfplumber (lepsza jakość ekstrakcji)
    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(pages).strip()
        if text:
            logger.info(f"sejm_search: PDF sparsowany (pdfplumber): {len(text)} znaków")
            return text[:max_chars]
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"sejm_search: pdfplumber błąd: {e}")

    # Próba 2: pypdf
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(pages).strip()
        if text:
            logger.info(f"sejm_search: PDF sparsowany (pypdf): {len(text)} znaków")
            return text[:max_chars]
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"sejm_search: pypdf błąd: {e}")

    logger.warning(f"sejm_search: brak biblioteki PDF (pdfplumber/pypdf) — zainstaluj jedną z nich")
    return None


# ============================================================
# Detekcja — czy pytanie wymaga aktualnych przepisów
# ============================================================

def needs_current_law(question: str) -> bool:
    """
    Heurystyka: czy pytanie wymaga sprawdzenia aktualnego stanu prawa.
    Strategia INKLUZJI: Sejm API odpytywane TYLKO gdy pytanie zawiera terminy prawne.
    """
    q_lower = question.lower()
    return any(signal in q_lower for signal in LEGAL_SIGNALS)


def extract_legal_keywords(question: str) -> str:
    """
    Ekstrakcja słów kluczowych do wyszukiwania w API Sejmu.

    Dwa kroki:
    1. legal_terms — terminy prawne z listy + frazy wielowyrazowe + art. N
    2. semantic_terms — słowa ≥4 znaki semantycznie kluczowe dla pytania

    Fixe względem v2.3:
    - "art. 3" (liczba po art.) zachowane w całości
    - "umowa o pracę" → "umowa pracę" (nie "umowa o" — "o" wypadało)
    - semantic_terms uzupełniają query o słowa których nie ma w legal_terms
    """
    q_lower = question.lower()

    # Krok 1a: frazy wielowyrazowe — muszą być przed single-word regex
    MULTI_WORD = [
        "kodeks pracy", "kodeks cywilny", "kodeks spółek handlowych",
        "umowa o pracę", "umowa zlecenia", "umowa o dzieło",
        "prawo pracy", "prawo cywilne",
        "minimalne wynagrodzenie", "wynagrodzenie minimalne",
        "składki zus", "ubezpieczenie społeczne",
        "czas pracy", "urlop wypoczynkowy", "urlop macierzyński",
        "wypowiedzenie umowy", "rozwiązanie umowy",
        "zakaz konkurencji", "klauzula poufności",
    ]
    found_multi = []
    q_for_single = q_lower
    for phrase in MULTI_WORD:
        if phrase in q_lower:
            found_multi.append(phrase)  # fraza ze spacją dla API
            # Usuń frazę z tekstu żeby single-word jej nie rozbił
            q_for_single = q_for_single.replace(phrase, "")

    # Krok 1b: art. N — zachowaj numer artykułu
    art_refs = re.findall(r'\bart\.?\s*(\d+[a-z]?)\b', q_lower)
    art_terms = [f"art.{n}" for n in art_refs]

    # Krok 1c: pojedyncze terminy prawne
    legal_terms = re.findall(
        r"\b(?:ustawa|kodeks|rozporządzenie|dyrektywa"
        r"|k\.p\.?|k\.c\.?|ksh|kpa|vat|pit|cit|zus"
        r"|pracownik|pracodawca|wynagrodzenie|urlop|zwolnienie"
        r"|najem|dzierżawa|sprzedaż|darowizna|testament"
        r"|podatek|składki?|ubezpieczenie|emerytura|renta"
        r"|zleceniobiorca|zleceniodawca|najemca|wynajmujący"
        r"|obowiązki|uprawnienia|odpowiedzialność|roszczenie"
        r"|klauzula|poufność|wypowiedzenie|odszkodowanie"
        r"|spółka|działalność|pełnomocnictwo|karencja|etat)\b",
        q_for_single
    )

    # Krok 2: semantic_terms — słowa ≥4 znaki poza stop-listą
    STOP_SEMANTIC = {
        "można", "należy", "trzeba", "jest", "będzie", "były", "przez",
        "zgodnie", "które", "który", "która", "jakie", "jaki", "jaka",
        "kiedy", "gdzie", "czego", "czemu", "proszę", "podaj", "powiedz",
        "określone", "określony", "dotyczy", "polska", "polsce", "roku",
        "obowiązujący", "obowiązującym", "aktualny", "aktualnym",
        "podstawie", "zakresie", "przypadku", "sytuacji", "wynika",
        "powinien", "powinna", "powinno", "należy", "trzeba",
    }
    semantic_terms = [
        w for w in re.findall(r'\b[a-złśźćńąęóż]{4,}\b', q_for_single)
        if w not in STOP_SEMANTIC
        and w not in legal_terms
        and w not in found_multi
    ]

    # Złóż: frazy wielowyrazowe mają priorytet, potem art. N, potem legal, potem semantic
    all_kw = found_multi + art_terms + legal_terms
    seen: set[str] = set(all_kw)
    slots_left = 8 - len(all_kw)
    for w in semantic_terms:
        if slots_left <= 0:
            break
        if w not in seen:
            seen.add(w)
            all_kw.append(w)
            slots_left -= 1

    # Deduplikacja z zachowaniem kolejności
    unique = []
    seen2: set[str] = set()
    for kw in all_kw:
        if kw not in seen2:
            seen2.add(kw)
            unique.append(kw)

    result = " ".join(unique[:8])
    logger.debug(f"sejm_search: słowa kluczowe: '{result}'")
    return result


# ============================================================
# Formatowanie kontekstu dla modeli
# ============================================================


def _score_chunk(chunk: str, keywords: list) -> float:
    """Scoring chunk po słowach kluczowych — pełne słowo = 2 pkt, fragment = 1 pkt."""
    chunk_lower = chunk.lower()
    score = 0.0
    for kw in keywords:
        kw_l = kw.lower()
        if kw_l in chunk_lower:
            score += 2.0 if re.search(r'\b' + re.escape(kw_l) + r'\b', chunk_lower) else 1.0
    return score


def _extract_relevant_chunks(
    text: str,
    keywords: list,
    max_chars: int = 4500,
) -> str:
    """
    Pobiera z tekstu aktu TYLKO artykuły trafne dla pytania.

    Zamiast "pierwsze N znaków" (co losowo odcina odpowiedź w połowie ustawy),
    dzieli tekst na artykuły, rankuje po słowach kluczowych z pytania,
    zwraca trafne artykuły + ich sąsiedzi (kontekst).

    Jeśli żaden artykuł nie pasuje — fallback do pierwszych 5 artykułów.
    """
    # Podziel na artykuły
    parts = re.split(r'(?=\n(?:Art\.|Artykuł)\s*\d+)', '\n' + text)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return text[:max_chars]

    scored = [((_score_chunk(p, keywords)), i, p) for i, p in enumerate(parts)]

    # Fallback: żadne słowo kluczowe nie trafia — zwróć pierwsze artykuły
    if not any(s > 0 for s, _, _ in scored):
        logger.debug("sejm_search: brak trafień kw — fallback do pierwszych artykułów")
        return '\n\n'.join(p for _, _, p in scored[:5])[:max_chars]

    # Wybierz trafne artykuły + sąsiedzi dla kontekstu
    selected: set = set()
    for score, idx, _ in sorted(scored, key=lambda x: -x[0]):
        if score < 1.0:
            break
        selected.add(idx)
        if idx > 0:
            selected.add(idx - 1)
        if idx < len(parts) - 1:
            selected.add(idx + 1)

    result = '\n\n'.join(parts[i] for i in sorted(selected))
    n_selected = len(selected)
    n_total = len(parts)
    logger.debug(f"sejm_search: wybrano {n_selected}/{n_total} artykułów po keyword scoring")
    return result[:max_chars]

def format_context_for_llm(
    results: SearchResult,
    include_text: bool = True,
) -> str:
    """
    Formatuje wyniki wyszukiwania jako blok kontekstu dla modeli.

    Jeśli include_text=True — pobiera tekst pierwszego obowiązującego aktu.
    Zawsze zwraca metadane (tytuł, data wejścia w życie, status, link ISAP).

    Zwracany string trafia do system prompt przed pytaniem użytkownika.
    """
    if results.error:
        return (
            f"[WYSZUKIWANIE PRZEPISÓW: błąd — {results.error}. "
            f"Odpowiedz na podstawie własnej wiedzy i zaznacz że nie udało się "
            f"potwierdzić aktualnego stanu prawa.]"
        )

    if not results.acts:
        return (
            f"[WYSZUKIWANIE PRZEPISÓW: brak wyników dla zapytania '{results.query}'. "
            f"Odpowiedz na podstawie własnej wiedzy i wyraźnie zaznacz datę odcięcia "
            f"wiedzy oraz że nie udało się znaleźć aktualnego aktu prawnego. "
            f"Zaproponuj weryfikację na isap.sejm.gov.pl]"
        )

    lines = [
        f"[AKTUALNE PRZEPISY — źródło: api.sejm.gov.pl, Kancelaria Sejmu RP]",
        f"Zapytanie: {results.query} | Znaleziono: {results.total} aktów\n",
    ]

    # Słowa kluczowe do scoringu artykułów — z query Sejmu
    _kw_list = results.query.split() if results.query else []

    for i, act in enumerate(results.acts[:3], 1):
        status_label = "✓ OBOWIĄZUJĄCY" if act.is_active else f"✗ {act.status.upper()}"
        lines.append(f"--- Akt {i}: {status_label} ---")
        lines.append(f"Tytuł:    {act.title}")
        lines.append(f"Adres:    {act.display_address}")
        if act.entry_into_force:
            lines.append(f"W życie:  {act.entry_into_force}")
        if act.keywords:
            lines.append(f"Słowa kluczowe: {', '.join(act.keywords[:5])}")
        lines.append(f"Źródło:   https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id="
                     f"WDU{act.year:04d}0000{act.position:04d}")

        if include_text and act.is_active:
            full_text = None
            source_label = ""

            if act.text_html:
                full_text = get_act_text(act.publisher, act.year, act.position)
                source_label = "HTML"
            
            if not full_text:
                # Fallback: PDF — dla rozporządzeń RM które nie mają HTML
                full_text = get_act_text_pdf(act.publisher, act.year, act.position)
                source_label = "PDF"

            if full_text:
                per_act_limit = MAX_TEXT_CHARS if i == 1 else 3000
                # Rozporządzenia RM są krótkie (3-5 paragrafów) — daj całość bez scoringu
                # Długie akty (ustawy) — keyword scoring żeby nie zalewać kontekstu
                if len(full_text) <= 2000:
                    lines.append(f"\nPEŁNY TEKST ({source_label}):")
                    lines.append(full_text[:per_act_limit])
                else:
                    relevant = _extract_relevant_chunks(full_text, _kw_list,
                                                        max_chars=per_act_limit)
                    lines.append(f"\nTRAFNE PRZEPISY (keyword scoring, {source_label}, max {per_act_limit} znaków):")
                    lines.append(relevant)
            else:
                lines.append("\n[Tekst niedostępny — brak HTML i PDF]")

        lines.append("")

    lines.append(
        "[INSTRUKCJA DLA MODELU: powyższe przepisy są aktualne na dziś. "
        "Analizuj dokument klienta WZGLĘDEM tego tekstu. "
        "Jeśli pytanie dotyczy interpretacji lub orzeczeń — zaznacz że "
        "interpretacje KIS i orzeczenia NSA wymagają osobnej weryfikacji "
        "na podatki.gov.pl i orzeczenia.nsa.gov.pl]"
    )

    return "\n".join(lines)


# ============================================================
# Główna funkcja — użycie w main.py
# ============================================================

def get_legal_context(question: str) -> Optional[str]:
    """
    Główna funkcja do wywołania z main.py.

    Zwraca string kontekstu do wstawienia w system prompt,
    lub None jeśli pytanie nie wymaga aktualnych przepisów.

    Użycie w main.py (przed wywołaniem modeli):

        from sejm_search import get_legal_context
        legal_ctx = get_legal_context(q.question)
        if legal_ctx:
            system_prompt = legal_ctx + "\\n\\n" + SYSTEM_PROMPT_GENERAL

    Nie rzuca wyjątków — błędy są zalogowane i zwracany jest None.
    """
    if not needs_current_law(question):
        return None

    keywords = extract_legal_keywords(question)
    if not keywords:
        return None

    logger.info(f"sejm_search: pytanie wymaga aktualnych przepisów, szukam: '{keywords}'")

    try:
        results = search_acts(keywords, in_force_only=True)
        # [FIX] Nie przekazuj komunikatu błędu do modeli jako "kontekst prawny".
        # SearchResult.error oznacza że API nie odpowiedziało — modele dostaną None
        # i odpowiedzą na podstawie własnej wiedzy, zaznaczając brak weryfikacji.
        if results.error:
            logger.warning(f"sejm_search: błąd API — nie wstrzykuję kontekstu: {results.error}")
            return None
        context = format_context_for_llm(results, include_text=True)
        return context
    except Exception as e:
        logger.error(f"sejm_search: get_legal_context błąd: {e}")
        return None