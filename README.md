# resume-jev-match

A small FastAPI service that scores a resume against a job description using
[TypeSafe AI's Jev model](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
called through LangChain's `langchain-typesafe` integration.

This is a learning/portfolio project, not a production system — the goal was
to understand how a "System One" model differs from a normal LLM call, and
how LangChain wraps it as a typed `Runnable`.

## What's a "System One" model, and how is Jev different from a normal LLM call?

A normal LLM call generates text, token by token, and your application then
has to parse that text back into something structured — hoping the model
formatted its JSON correctly, didn't hedge, and didn't hallucinate a field
that doesn't exist.

Jev skips that round trip. You send it a piece of **state** (plain text or
JSON) plus a set of **typed questions**, and it answers all of them in a
single parallel pass — in 70-500ms — with:

- a **probability** (`Noul` — "does the resume mention Kubernetes?" → `0.83`)
- a **label with a probability distribution** (`Choice` — pick one of a fixed
  set of options, each with its own probability, plus an overall confidence)
- a **position on an ordered scale** (`Score` — a probability-weighted mean
  across ordered rubric levels, e.g. "how severe is this?")

There's no text to parse and no way for the model to return a value outside
the schema you declared — it can't invent a `Choice` label that wasn't in
your `criteria`. In exchange for that structure, Jev doesn't do open-ended
generation: it's built specifically for decisions, not conversation. The
name comes from Kahneman's *Thinking, Fast and Slow* — Jev is "System 1",
fast pattern-matching, not "System 2" deliberate reasoning.

## Why `TypeSafeClassifier` instead of the raw TypeSafe SDK?

- It's a LangChain `Runnable`, so it gets `.invoke()`, `.batch()`, and
  `.ainvoke()` for free, and composes with the rest of LangChain the same
  way a chat model does.
- It auto-traces to [LangSmith](https://smith.langchain.com) when
  configured, so every classification — its state, its questions, its
  answers, token usage — shows up in your trace viewer without extra code.
- Retries with exponential backoff on `429`/`529`/connection failures are
  built into the underlying SDK by default; you don't have to hand-roll
  that.
- Its errors (`TypeSafeRateLimitError`, `TypeSafeAuthenticationError`, etc.)
  inherit from LangChain's standard model-error hierarchy, so code that
  already handles LangChain provider errors uniformly keeps working.

## What this service judges

Given one resume and one job description, in a single fan-out request:

| Question | Type | What it answers |
|---|---|---|
| `overall_fit` | `Score` | Where the candidate lands on a 3-level rubric, from "no meaningful overlap" to "strong match" |
| `has_required_skills` | `Noul` | Does the resume mention the JD's core technical skills? |
| `seniority_match` | `Choice` | underqualified / matched / overqualified / unclear |

All three are asked together in one `.invoke()` call — Jev evaluates
independent questions in parallel within a single request, so this is both
cheaper and faster than three sequential calls.

**Design choices worth knowing about**, since they diverge slightly from
"the obvious thing":

- `seniority_match` has a 4th option, `unclear`, beyond the three the spec
  named. Jev's own guidance is to always give `Choice` an escape hatch when
  the label set might not cover every input — here, that's a resume/JD pair
  too thin to judge seniority from.
- `needs_human_review` is `true` when the *lower* of `fit_confidence` and
  `seniority_confidence` falls below a threshold (default `0.5`, tunable via
  `CONFIDENCE_GATE_THRESHOLD`). `has_required_skills` (a `Noul` answer) has
  no confidence field by design — Jev returns the probability itself instead
  — so it can't drive this gate.
- `fit_score` in the API response is normalized to `0.0-1.0`. Jev's raw
  `Score.score` is a probability-weighted mean of the rubric's zero-based
  *level index* (so `0` to `2` for a 3-level rubric), not already a
  probability.

## Project layout

```
app/
  classifier.py   # the Jev/LangChain integration — no FastAPI dependency
  config.py       # typed settings (pydantic-settings), read once from env
  models.py       # request/response schemas for the HTTP API
  main.py         # FastAPI app: /match, /batch-match, /health
data/
  resumes/            # 6 sample resumes, varied fit levels
  job_descriptions/   # 3 sample job descriptions
tests/            # classifier + API tests, TypeSafeClassifier mocked out
```

`classifier.py` is deliberately independent of FastAPI: it's the reusable,
directly-testable core, and `main.py` just wires it up to HTTP.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # or requirements-dev.txt for tests/lint/type-check

cp .env.example .env
# then edit .env and set TYPESAFE_API_KEY
```

Get a key from [TypeSafe's waitlist](https://typesafe.ai), or skip the wait
by routing through the
[Vercel AI Gateway](https://vercel.com/docs/ai-gateway) instead:

```bash
TYPESAFE_API_KEY=<your Vercel AI Gateway credential>
TYPESAFE_BASE_URL=https://ai-gateway.vercel.sh/typesafe
```

## Run

```bash
uvicorn app.main:app --reload
```

Or with Docker:

```bash
docker build -t resume-jev-match .
docker run --env-file .env -p 8000:8000 resume-jev-match
```

Interactive API docs are at `http://localhost:8000/docs`.

## Example requests

**Single match:**

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "resume": "Senior backend engineer, 6 years Python, FastAPI, PostgreSQL, AWS, Kafka...",
  "job_description": "Senior Backend Engineer — Python, FastAPI, PostgreSQL, AWS, Kafka..."
}
EOF
```

```json
{
  "fit_score": 1.0,
  "fit_confidence": 0.94,
  "fit_legend": "Strong match — most required skills and experience present",
  "has_required_skills": true,
  "has_required_skills_probability": 0.97,
  "seniority_match": "matched",
  "seniority_confidence": 0.88,
  "model": "jev-1.13.0",
  "needs_human_review": false
}
```

**Batch match** (using the sample data in `data/`):

```bash
curl -X POST http://localhost:8000/batch-match \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "resumes": [
    {"id": "senior-strong", "resume": "..."},
    {"id": "junior-partial", "resume": "..."},
    {"id": "frontend-nomatch", "resume": "..."}
  ],
  "job_description": "..."
}
EOF
```

The `results` array comes back sorted by `fit_score`, descending.

## Testing

```bash
pip install -r requirements-dev.txt
ruff check .      # lint
mypy app          # type check
pytest            # unit + API tests, TypeSafeClassifier mocked out — no network calls
```

## What I learned

- A "typed" model response is a different contract than "structured output"
  from a normal LLM. Structured-output mode still generates text under the
  hood and validates it after the fact; Jev's schema is enforced by
  construction, so there's no failure mode where the model "almost" returns
  valid JSON.
- `Score.legend` comes back as the *entire* rubric (`{0: "...", 1: "...",
  2: "..."}`), not just the description nearest the answer — you pick the
  closest level yourself from the (possibly fractional) score. I initially
  assumed the SDK would hand back a single matched string; reading the
  actual response type caught that before it shipped.
- Confidence and probability aren't the same axis: `Choice.confidence`
  describes how concentrated the whole probability distribution is, not how
  likely the *selected* label is. A `Noul` question, by contrast, has no
  confidence field at all — the probability *is* the answer. That asymmetry
  is why the confidence gate here can't treat all three answer types
  uniformly.
- Fan-out (asking every question in one request) isn't just a latency
  optimization — it also gives you results that are consistent with each
  other, since they're all conditioned on the same evaluation pass instead
  of three independent ones.
- Mocked tests weren't enough to catch everything: `TypeSafeClassifier()`
  reads `TYPESAFE_API_KEY` from the real process environment, but loading a
  `.env` file through `pydantic-settings` only populates *our* `Settings`
  object — it doesn't export the value into `os.environ`. `/health` looked
  fine (it only checks `Settings`), but the first real request against the
  live API failed until `get_classifier()` was changed to pass
  `settings.typesafe_api_key` into `TypeSafeClassifier` explicitly instead
  of relying on it to re-read the environment on its own.
