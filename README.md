# Triangulum

Triangulum is a small experimental PWA built to test a simple idea: one strong model is useful, but several models with different roles can be more reliable when their disagreements are exposed instead of averaged away.

The application sends the same question to multiple frontier models, compares their answers, looks for agreement and contradictions, and then produces a synthesis. The current implementation also includes online verification paths, fallback handling, cost limits, caching and backend checks around model behavior.

This is not a research benchmark and it is not presented as proof that multi-model systems are always better. It is a practical project built through repeated failures, corrections and redesigns.

## Why I built it

I am not a software engineer or ML engineer. I am an independent AI practitioner who uses models to build and test real workflows.

While working with coding and reasoning models I kept seeing the same problem: a model could produce a fluent answer, approve its own work, and still miss the actual requirement. I started separating roles instead of asking one model to do everything. One model could produce an answer, another could challenge it, while deterministic backend rules handled things that should not depend on model memory or judgment.

Triangulum grew out of that process.

## What the system does

```text
User question
     |
     v
Multiple independent model answers
     |
     v
Comparison and verification
     |
     v
Blind synthesis / falsification
     |
     v
Backend guards, limits and fallback logic
     |
     v
User report
```

The current code integrates models from Anthropic, OpenAI and Google. Some verification paths also use external sources such as Brave Search and Polish parliamentary data.

A recent change replaced a simple majority-style synthesis with a blind comparison. Raw answers are shuffled and shown to the synthesis model as A, B and C so the judge does not know which provider produced which answer. Minority positions are preserved instead of being discarded automatically.

## What I tested and learned

The most useful part of this project was not producing the first working version. It was finding where the system failed.

Examples include:

- a model ignoring fresh web context when it conflicted with its prior knowledge
- stream and non-stream paths drifting into different behavior
- a synthesis step losing minority claims before they could be evaluated
- self-preference risk when a model could identify its own provider's answer
- fallback behavior silently hiding degraded operation
- one slow model consuming the time budget of faster models
- prompt injection risks through uploaded files or carried context
- model output changing pseudonymization token formats
- a technically successful response still violating the intended contract

These cases were turned into backend fixes, tighter prompts, explicit guards or regression checks. A longer record is in [docs/FAILURE_LOG.md](docs/FAILURE_LOG.md).

## Evaluation approach

The project uses several evaluation patterns that also appear in professional LLM evaluation work:

- side-by-side comparison of model outputs
- explicit specification and instruction adherence checks
- recurring failure-mode analysis
- adversarial and edge-case testing
- grounding and hallucination checks
- human-in-the-loop decisions instead of automatic trust in consensus
- deterministic backend controls for rules that should not be delegated to an LLM
- regression-oriented fixes after discovered failures

I do not claim commercial RLHF, SFT or model-training experience. This is self-directed, project-based work.

## My role and the role of AI

The project was built through a human-AI development workflow.

My role was to define the product behavior, decide how models should be separated into roles, test outputs, identify failures, set acceptance conditions and decide which fixes were acceptable. AI coding agents generated and revised much of the implementation.

That distinction matters to me. I use AI to write code I could not write independently, but I remain responsible for the workflow, tests, decisions and evaluation of whether the result matches the intended behavior.

## Stack

Python, FastAPI, React PWA, SQLite, Anthropic API, OpenAI API, Google GenAI, Brave Search and Polish Sejm data sources.

## Running the project

The public deployment uses environment variables for provider credentials and operational settings. No production API keys are intended to be stored in this repository.

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the service:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The application requires valid API credentials for the model providers used by the selected path.

## Status

Experimental personal project. The repository contains the working application and also reflects the history of iterative AI-assisted development. It is intentionally presented as a case study in model behavior, evaluation and control rather than as a polished software-engineering showcase.
