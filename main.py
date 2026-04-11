import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional

from openai import OpenAI
from anthropic import Anthropic
from google import genai as google_genai

# =======================
# Init
# =======================
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_api_key = os.getenv("GEMINI_API_KEY")


# =======================
# Schemas
# =======================
class Question(BaseModel):
    question: str

class ChatMessage(BaseModel):
    question: str
    context: Optional[str] = ""
    history: Optional[list] = []


# =======================
# Model calls
# =======================
def ask_openai(question: str) -> str:
    try:
        r = openai_client.responses.create(
            model="gpt-4o-mini",
            input=question
        )
        return r.output[0].content[0].text
    except Exception as e:
        return f"[OPENAI ERROR] {e}"


def ask_claude(question: str) -> str:
    try:
        r = claude_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": question}]
        )
        return r.content[0].text
    except Exception as e:
        return f"[CLAUDE ERROR] {e}"


def ask_gemini(question: str) -> str:
    if not gemini_api_key:
        return "[GEMINI: brak klucza API]"
    try:
        client = google_genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=question
        )
        return response.text
    except Exception as e:
        return f"[GEMINI ERROR] {e}"


# =======================
# Synthesis - SILNIK
# =======================
def synthesize(question: str, a: str, b: str, c: str = "") -> str:
    gemini_section = f"Odpowiedź C (Gemini):\n{c}" if c and not c.startswith("[GEMINI") else ""
    prompt = f"""Piszesz odpowiedź dla zwykłego człowieka który chce zrozumieć temat.

Ton: jak mądry znajomy który zna się na rzeczy. Rzeczowy, bezpośredni, ciepły.
Język: prosty polski. Krótkie zdania. Każde musi coś mówić.
Jeśli pytanie zawiera fałszywe założenie - powiedz to wprost na początku.
Nawet jeśli temat jest subiektywny — zawsze podaj konkretne przykłady, nazwy, tytuły lub osoby które oba modele wymieniły. Nie zostawiaj użytkownika bez żadnej konkretnej odpowiedzi.
Bądź szczodry w szczegółach — użytkownik chce dowiedzieć się czegoś nowego, nie tylko usłyszeć że modele się zgadzają. Rozwiń każdy punkt raportu tak żeby użytkownik wychodził z realną wiedzą.

Masz odpowiedzi modeli na pytanie: {question}

Odpowiedź A (Claude):
{a}

Odpowiedź B (GPT-4o-mini):
{b}

{gemini_section}

Napisz raport w tej strukturze:

**SYNTEZA:** [1-2 zdania - prosto: tak / nie / zależy od czego]

**CO Z TEGO WYNIKA**
[Konkretna odpowiedź - co powinien wiedzieć użytkownik]

**DLACZEGO TAK**
[Wyjaśnienie mechanizmu prostym językiem]

**TWARDE FAKTY**
[2-4 konkretne liczby lub przykłady - tylko jeśli są w materiale. Jeśli brak: napisz "Brak twardych danych w tym pytaniu."]

**CO WIEMY, A CZEGO NIE**
- pewne: [co jest dobrze potwierdzone]
- częściowe: [co jest mniej jasne]
- niepewne: [co jest spekulacją]

**GDZIE SĄ GRANICE**
[Co zależy od sytuacji, czego system nie może rozstrzygnąć]"""

    try:
        r = openai_client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=2000
        )
        return r.output[0].content[0].text
    except Exception as e:
        return f"[SYNTHESIS ERROR] {e}"


# =======================
# Adam - konwersacja
# =======================
def adam_reply(question: str, context: str, history: list) -> str:
    system = """Jesteś Adamem - asystentem który pomaga użytkownikowi pogłębić temat.

Masz dostęp do raportu który system już wygenerował. Twoim zadaniem jest rozwinąć konkretny fragment, wyjaśnić mechanizm lub odpowiedzieć na pytanie pogłębiające.

Zasady:
- odpowiadasz tylko na podstawie kontekstu z raportu i swojej wiedzy
- nie zmieniasz wniosków raportu
- jeśli pytanie wykracza poza raport - mówisz to wprost i sugerujesz nowe zapytanie do SILNIKA
- ton: naturalny, rzeczowy, ciepły
- długość: tyle ile potrzeba, nie więcej
- język: prosty polski"""

    messages = []

    if context:
        messages.append({
            "role": "user",
            "content": f"Oto raport który wygenerowałem wcześniej:\n\n{context}"
        })
        messages.append({
            "role": "assistant",
            "content": "Rozumiem. Mam ten raport przed sobą. O co chcesz zapytać?"
        })

    for msg in history:
        messages.append(msg)

    messages.append({
        "role": "user",
        "content": question
    })

    try:
        r = claude_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=800,
            system=system,
            messages=messages
        )
        return r.content[0].text
    except Exception as e:
        return f"[ADAM ERROR] {e}"


# =======================
# API
# =======================

@app.post("/ask")
def ask(q: Question):
    try:
        with ThreadPoolExecutor(max_workers=3) as ex:
            fa = ex.submit(ask_openai, q.question)
            fb = ex.submit(ask_claude, q.question)
            fc = ex.submit(ask_gemini, q.question)

            try:
                a = fa.result(timeout=30)
            except TimeoutError:
                a = "[OPENAI TIMEOUT]"

            try:
                b = fb.result(timeout=30)
            except TimeoutError:
                b = "[CLAUDE TIMEOUT]"

            try:
                c = fc.result(timeout=30)
            except TimeoutError:
                c = "[GEMINI TIMEOUT]"

        s = synthesize(q.question, a, b, c)

        return {
            "question": q.question,
            "openai": a,
            "claude": b,
            "gemini": c,
            "synthesis": s,
            "status": "ok"
        }

    except Exception as e:
        return {
            "question": q.question,
            "openai": "",
            "claude": "",
            "gemini": "",
            "synthesis": "",
            "status": "error",
            "error": str(e)
        }


@app.post("/chat")
def chat(msg: ChatMessage):
    try:
        answer = adam_reply(
            question=msg.question,
            context=msg.context,
            history=msg.history
        )
        return {
            "answer": answer,
            "status": "ok"
        }
    except Exception as e:
        return {
            "answer": "",
            "status": "error",
            "error": str(e)
        }


@app.get("/")
def root():
    html = """<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>SILNIK v3</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0c0f;color:#c8cdd6;font-family:'IBM Plex Sans',sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}
.box{max-width:480px;width:100%;text-align:center}
.label{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:.2em;color:#2d4a6e;text-transform:uppercase;margin-bottom:8px}
.title{font-size:22px;font-weight:300;color:#e0e6ee;letter-spacing:-.02em;margin-bottom:4px}
.title span{font-weight:600;color:#5b9bd5}
.info{font-size:13px;color:#4a5560;margin:20px 0 28px;line-height:1.7}
.step{background:#0d1015;border:1px solid #1a1f26;border-radius:3px;padding:14px 18px;margin-bottom:10px;text-align:left}
.step-label{font-family:'IBM Plex Mono',monospace;font-size:9px;color:#2d4a6e;letter-spacing:.15em;text-transform:uppercase;margin-bottom:6px}
.step-text{font-size:13px;color:#8a96a4;line-height:1.6}
.step-code{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#5b9bd5;background:#080b0e;padding:4px 8px;border-radius:2px;margin-top:6px;display:inline-block}
.status{font-family:'IBM Plex Mono',monospace;font-size:10px;color:#3a7a5a;margin-top:20px;letter-spacing:.08em}
</style>
</head>
<body>
<div class="box">
  <div class="label">System Weryfikacji Multi-Model</div>
  <div class="title">SILNIK <span>v3.0</span></div>
  <div class="info">Backend działa. Aby użyć interfejsu, otwórz plik <strong style="color:#c8cdd6">silnik_v3.jsx</strong> w Claude.ai jako artefakt.</div>
  <div class="step">
    <div class="step-label">Endpoint SILNIK</div>
    <div class="step-text">Zapytania analityczne z dwoma modelami</div>
    <div class="step-code">POST /ask</div>
  </div>
  <div class="step">
    <div class="step-label">Endpoint ADAM</div>
    <div class="step-text">Pogłębiona rozmowa na podstawie raportu</div>
    <div class="step-code">POST /chat</div>
  </div>
  <div class="step">
    <div class="step-label">Dokumentacja API</div>
    <div class="step-text">Interaktywny interfejs do testowania</div>
    <div class="step-code"><a href="/docs" style="color:#5b9bd5;text-decoration:none">http://127.0.0.1:8000/docs</a></div>
  </div>
  <div class="status">✓ SILNIK aktywny · Claude + GPT-4o-mini</div>
</div>
</body>
</html>"""
    return Response(content=html, media_type="text/html")