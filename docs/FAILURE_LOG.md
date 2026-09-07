# Failure log

This file records selected failures that changed how Triangulum was designed. It is not a complete changelog. The purpose is to show what went wrong, how the failure was recognized and what kind of control was added afterward.

## Fresh context ignored by the model

Observed behavior: a model could prefer its internal knowledge over fresher web or legal context supplied in the prompt.

Why it mattered: a fluent answer could look correct while being stale or directly inconsistent with current evidence.

Response: context handling was tightened so current web and legal material is passed explicitly and models are instructed not to guess current facts from training knowledge when fresh context exists.

Evaluation lesson: grounding should be checked as behavior, not assumed from prompt presence.

## Stream and non-stream behavior drift

Observed behavior: duplicated prompt logic in separate execution paths could evolve differently.

Why it mattered: the same user request could be evaluated or synthesized differently depending on transport path rather than task content.

Response: synthesis prompt construction was moved into shared logic used by both paths.

Evaluation lesson: behavior parity across execution paths needs regression checks. Functional duplication can become model-behavior drift.

## Majority agreement hid useful minority claims

Observed behavior: an earlier synthesis path reduced model answers to aligned facts, contradictions and uncertain claims before the final synthesis. Minority claims could lose their original context before the judge saw them.

Why it mattered: two models agreeing does not prove they are correct. A single model may contain the strongest answer.

Response: the synthesis model now receives the raw answers and is instructed to preserve and assess minority positions instead of automatically averaging them away.

Evaluation lesson: consensus is evidence, not ground truth.

## Potential judge self-preference

Observed risk: if the synthesis model knows which answer came from its own model family, provider identity may influence judgment.

Response: raw model answers are shuffled and presented to the synthesis step as Answer A, B and C. Provider labels are hidden during judging.

Evaluation lesson: side-by-side evaluation is stronger when irrelevant identity cues are removed.

## Slow model consumed shared time budget

Observed behavior: sequential waiting could allow one slow provider to consume time that should still be available to faster providers.

Response: parallel execution was changed to absolute deadlines and provider-specific hard timeouts.

Evaluation lesson: agent or model evaluation includes orchestration behavior. A correct component can still damage system reliability through timing.

## Silent degradation in fallback paths

Observed behavior: a provider or grounding dependency could fail while the application continued through a fallback path with insufficiently visible degradation.

Response: fallback checks, partial-model annotations and degraded-mode warnings were added in several paths.

Evaluation lesson: graceful degradation should remain observable. A successful HTTP response is not the same as a successful AI task.

## Prompt injection through files and carried context

Observed risk: uploaded documents or previously generated context could be inserted into higher-trust prompt locations.

Response: uploaded and carried content was sandboxed, length-limited or moved into explicit context blocks; additional prompt-injection checks were added around input paths.

Evaluation lesson: trust boundaries matter in model workflows. User-controlled text should not silently become system-level instruction.

## Model changed pseudonymization token format

Observed behavior: model output could alter placeholder formatting, for example removing underscores from generated tokens.

Why it mattered: downstream deterministic logic depended on exact token identity.

Response: detection logic was expanded to recognize altered forms rather than assuming the model would preserve formatting exactly.

Evaluation lesson: do not rely on an LLM to preserve machine-significant syntax unless it is verified after generation.

## Backend rules were safer than repeated prompting

Observed pattern: some failures kept returning when control depended only on prompt instructions.

Response: critical requirements were progressively moved into deterministic mechanisms such as hard timeouts, authentication checks, budget gates, caching rules, output checks and shared backend logic.

Evaluation lesson: model instructions and deterministic controls solve different classes of problems. Important invariants should be enforced outside the model whenever possible.

## A technical PASS could still miss the intended behavior

Observed pattern: an implementation could run, pass local checks and still violate the intended product contract or architecture.

Response: evaluation was separated from implementation. Acceptance conditions, independent review and human approval became separate stages rather than asking the producing model to certify its own work.

Evaluation lesson: self-evaluation is useful evidence, but it is not independent validation.

## Current open questions

Triangulum is still experimental. The project does not establish that multi-model workflows always outperform a single strong model. Questions still worth testing include:

- when independent models add useful diversity and when they only add correlated noise
- when blind judging actually reduces bias enough to justify extra cost
- how often minority answers are correct
- how stable judgments are across repeated runs
- which checks belong in model prompts and which should always be deterministic
- whether the system improves accuracy enough to justify latency and token cost

These are treated as evaluation questions, not as solved claims.
