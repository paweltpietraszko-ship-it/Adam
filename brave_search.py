# v2.0 — blacklist zamiast whitelist, relevance scoring, lazy API key
"""
Triangulum — brave_search.py  v2
=================================
Wstrzykuje aktualny kontekst z internetu przed triangulacją modeli.
Brave Search API — surowe wyniki, nie AI-summary.

v2 vs v1:
  - Whitelist zastąpiony blacklistą: więcej trafnych źródeł (pip.gov.pl,
    infor.pl, prawo.pl, lex.pl nie wymagają whitelistowania)
  - Relevance scoring snippetów: wyniki rankowane po słowach kluczowych,
    nie po kolejności Brave — trafniejszy kontekst przy mniejszej liczbie tokenów
  - Lazy read klucza API: czytany przy wywołaniu, nie przy imporcie modułu

Zasady:
  - Blacklist: social media, fora, blogi, zakupy, AI-farmy treści
  - Tylko słowa kluczowe z pytania do API — nigdy dane osobowe
  - Max 5 wyników do scoringu, top 3 do kontekstu, max 400 znaków snippetu
  - Fallback: brak kontekstu = triangulacja bez weba (nie blokuje)

Konfiguracja w .env:
  BRAVE_API_KEY=twój_klucz
  BRAVE_ENABLED=true
  BRAVE_MAX_RESULTS=3
  BRAVE_TIMEOUT=5
"""

import os
import re
import logging
import urllib.request
import urllib.parse
import json
from typing import Optional

logger = logging.getLogger("triangulum.brave")

# ============================================================
# Konfiguracja
# ============================================================

BRAVE_ENABLED     = os.getenv("BRAVE_ENABLED", "true").lower() == "true"
BRAVE_MAX_RESULTS = int(os.getenv("BRAVE_MAX_RESULTS", "3"))
BRAVE_TIMEOUT     = int(os.getenv("BRAVE_TIMEOUT", "5"))
BRAVE_FETCH_N     = BRAVE_MAX_RESULTS * 4  # pobierz więcej, rankuj, zwróć top N

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


def _get_brave_api_key() -> str:
    """Lazy read — zawsze świeży klucz, działa nawet gdy load_dotenv() po imporcie."""
    return os.getenv("BRAVE_API_KEY", "")


# ============================================================
# Blacklista domen — tylko oczywisty szum
# ============================================================

DOMAIN_BLACKLIST: set = {
    # Social media
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "youtube.com", "linkedin.com", "pinterest.com",
    # Polskie fora i social
    "reddit.com", "wykop.pl", "quora.com",
    # Zakupy i ogłoszenia
    "allegro.pl", "olx.pl", "ceneo.pl", "amazon.com", "ebay.com",
    "otomoto.pl", "otodom.pl",
    # Znane farmy kontent-SEO (bez wartości merytorycznej)
    "eporady24.pl", "money.pl/blog", "porady123.pl",
}


def _extract_domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _is_blocked(url: str) -> bool:
    """True jeśli domena na blackliście."""
    domain = _extract_domain(url)
    return any(domain == b or domain.endswith("." + b) for b in DOMAIN_BLACKLIST)


# ============================================================
# Ekstrakcja słów kluczowych (bez PII)
# ============================================================

def _extract_keywords(question: str) -> str:
    """
    Wyciąga słowa kluczowe z pytania — NIGDY dane osobowe.
    Usuwa tokeny anonimizacji, liczby (mogą być PESEL/NIP), krótkie słowa.
    """
    text = re.sub(r'\b(?:FIRMA|OSOBA|NUMER|KWOTA|ADRES)_\d{3}\b', '', question)
    text = re.sub(r'\b\d+\b', '', text)
    words = re.findall(r'[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{4,}', text)
    return " ".join(words[:15])


# ============================================================
# Relevance scoring
# ============================================================

def _score_result(result: dict, keywords: list) -> float:
    """
    Scoruje wynik Brave po słowach kluczowych.
    Tytuł wagowany 3x, snippet 1x — tytuł lepiej charakteryzuje dokument.
    """
    title   = result.get("title", "").lower()
    snippet = result.get("description", "").lower()
    extra   = " ".join(result.get("extra_snippets", [])).lower()
    full    = snippet + " " + extra

    score = 0.0
    for kw in keywords:
        kw_l = kw.lower()
        # Pełne słowo = 2 pkt, fragment = 1 pkt; tytuł 3x
        if re.search(r'\b' + re.escape(kw_l) + r'\b', title):
            score += 6.0
        elif kw_l in title:
            score += 3.0
        if re.search(r'\b' + re.escape(kw_l) + r'\b', full):
            score += 2.0
        elif kw_l in full:
            score += 1.0

    return score


def _best_snippet(result: dict, keywords: list, max_chars: int = 400) -> str:
    """
    Wybiera najbardziej trafne zdanie ze snippetu i extra_snippets.
    Jeśli żadne nie trafia — zwraca cały opis do max_chars.
    """
    desc   = result.get("description", "").strip()
    extras = result.get("extra_snippets", [])
    all_sentences = re.split(r'(?<=[.!?])\s+', desc)
    for e in extras:
        all_sentences += re.split(r'(?<=[.!?])\s+', e.strip())

    kw_lower = [k.lower() for k in keywords]

    def sent_score(s: str) -> int:
        sl = s.lower()
        return sum(1 for k in kw_lower if k in sl)

    scored = sorted(all_sentences, key=sent_score, reverse=True)
    best = scored[0] if scored else desc

    # Jeśli najlepsze zdanie jest puste lub za krótkie — fallback do opisu
    if len(best.strip()) < 30:
        best = desc

    return best[:max_chars]


# ============================================================
# Główna funkcja
# ============================================================

def get_web_context(question: str) -> Optional[str]:
    """
    Pobiera kontekst webowy dla pytania.

    Zwraca sformatowany string z trafnymi fragmentami lub None jeśli:
    - brak klucza API / wyłączony
    - za mało słów kluczowych
    - błąd sieciowy
    - brak wyników po odfiltrowaniu blacklisty

    BEZPIECZEŃSTWO: do API trafiają tylko słowa kluczowe, nigdy dane osobowe.
    """
    api_key = _get_brave_api_key()
    if not api_key or not BRAVE_ENABLED:
        logger.debug("[BRAVE] Brak klucza API lub wyłączony")
        return None

    keywords_str = _extract_keywords(question)
    keywords     = keywords_str.split()
    if len(keywords) < 2:
        logger.debug(f"[BRAVE] Za mało słów kluczowych: '{keywords_str}'")
        return None

    logger.info(f"[BRAVE] Szukam: '{keywords_str[:80]}'")

    try:
        params = urllib.parse.urlencode({
            "q":             keywords_str,
            "count":         BRAVE_FETCH_N,
            "country":       "PL",
            "search_lang":   "pl",
            "freshness":     "py",
            "result_filter": "web",
            "extra_snippets": "true",
        })
        req = urllib.request.Request(
            f"{BRAVE_API_URL}?{params}",
            headers={
                "Accept":               "application/json",
                "Accept-Encoding":      "gzip",
                "X-Subscription-Token": api_key,
            }
        )
        with urllib.request.urlopen(req, timeout=BRAVE_TIMEOUT) as resp:
            raw = resp.read()
            if resp.info().get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            data = json.loads(raw.decode("utf-8"))

    except urllib.error.HTTPError as e:
        logger.warning(f"[BRAVE] HTTP {e.code}: {e.reason}")
        return None
    except Exception as e:
        logger.warning(f"[BRAVE] Błąd połączenia: {e}")
        return None

    results = data.get("web", {}).get("results", [])

    # Filtruj blacklistę, scoruj, rankuj
    scored = []
    for r in results:
        url = r.get("url", "")
        if _is_blocked(url):
            logger.debug(f"[BRAVE] Blacklista: {url[:60]}")
            continue
        score = _score_result(r, keywords)
        scored.append((score, r))

    if not scored:
        logger.info("[BRAVE] Brak wyników po odfiltrowaniu blacklisty")
        return None

    # Sortuj malejąco po score, weź top N
    scored.sort(key=lambda x: -x[0])
    top = scored[:BRAVE_MAX_RESULTS]

    logger.info(f"[BRAVE] {len(scored)} wyników po blackliście, "
                f"top {len(top)} po scoringu")

    lines = [
        "AKTUALNY KONTEKST Z INTERNETU:",
        "Fragmenty rankowane po trafności dla pytania — użyj jako tło, "
        "nie jako jedyne źródło prawdy:\n",
    ]
    for i, (score, r) in enumerate(top, 1):
        title   = r.get("title", "").strip()
        url     = r.get("url", "")
        snippet = _best_snippet(r, keywords)
        domain  = _extract_domain(url)
        lines.append(f"[{i}] {title}")
        lines.append(f"    Źródło: {domain} ({url[:80]})")
        lines.append(f"    {snippet}\n")

    lines.append(
        "UWAGA: Kontekst z wyszukiwania — weryfikuj z oficjalnymi źródłami."
    )

    return "\n".join(lines)


# ============================================================
# Test standalone
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from dotenv import load_dotenv
    load_dotenv()

    tests = [
        "Jakie są zasady oskładkowania umowy zlecenia przy zbiegu z etatem?",
        "Jaka jest pogoda w Warszawie dzisiaj?",
        "Kurs euro do złotego dzisiaj",
    ]
    for q in tests:
        print(f"\n{'='*60}")
        print(f"PYTANIE: {q}")
        result = get_web_context(q)
        if result:
            print(result[:600])
        else:
            print("None — brak klucza lub wyników")
