# Aura Skincare: Aria, AI Voice Support Agent

Browser-based voice customer support agent for a fictional D2C skincare brand. Click **Start Call**,
speak to Aria, and get spoken answers about shipping, returns, cancellations, COD and live order status.

**Use Chrome or Edge** (the free Web Speech API is not available in Firefox).

- Live app: `<add your Render URL>`
- Demo video: `<add link>`
- Note: the free Render tier sleeps when idle, so the first load can take 30-60 seconds.

## Architecture

```
Mic -> Browser STT (Web Speech API, en-IN)
    -> POST /api/chat (text)
    -> FastAPI: Llama 3.3 70B on Groq with tool calling
         tools: get_order_details, check_eligibility, cancel_order
    -> reply text
    -> Browser TTS (speechSynthesis, en-IN voice) -> Speaker
```

| Layer | Choice |
|---|---|
| Frontend | Vanilla HTML/CSS/JS, no build step (`static/`) |
| Backend | Python, FastAPI (also serves the frontend, so one deploy and one HTTPS URL) |
| STT / TTS | Browser Web Speech API / `speechSynthesis` |
| LLM | Llama 3.3 70B via Groq free tier |

**Guardrails:** the system prompt holds the brand policies and rules (no promises outside policy, refuse
out-of-scope requests, never invent order data). Cancel, return and damage eligibility is computed in code
(`data.py: check_eligibility`), so policy answers are deterministic rather than left to the LLM.

**Tool use:** the LLM is given function schemas and decides when to call them. The backend runs the tool,
feeds the result back, and loops (max 4 rounds) until the LLM returns a spoken reply.

**Graceful degradation:** empty or garbled speech is re-prompted without an LLM call; invalid or missing
order IDs return structured errors that the LLM turns into a polite re-ask; LLM failures are retried once,
then Aria asks the customer to repeat; a failed summary falls back to a safe default.

**Post-call:** `/api/summarize` returns the chronological transcript plus a JSON outcome
(`customer_intent`, `order_id`, `resolution_status`, `call_summary`) validated with Pydantic.

## Project structure

```
main.py            FastAPI app, system prompt, tool loop, API endpoints
data.py            mock orders, policy logic, tool functions and schemas
static/            index.html, style.css, app.js (voice loop and UI)
tests/             offline tests (LLM mocked)
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                                  # add your free key from console.groq.com/keys
uvicorn main:app --reload
```
Open http://localhost:8000 in Chrome. Tests: `pip install -r requirements-dev.txt && python -m pytest -q tests`

## Test orders

| Order | Product | Status |
|---|---|---|
| ORD-101 | Vitamin C Serum, Rs 699 | Out for Delivery (BlueDart BD-982103, by 6 PM) |
| ORD-102 | Hydrating Sunscreen SPF 50, Rs 499 | Delivered 14 days ago |
| ORD-103 | Green Tea Face Wash + Toner, Rs 850 | Processing (cancellable) |

Cancellations are stored per session, so cancelling ORD-103 never affects the next tester.

## Deploy (Render)

Build: `pip install -r requirements.txt`. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
Set `GROQ_API_KEY` in the environment. `render.yaml` is included.

## Tell us how you think

> DRAFT: rewrite these in your own words before submitting.

**1. Why this architecture and stack?**
A modular pipeline (STT, LLM with tools, TTS) is easy to debug and swap per layer. The browser handles STT
and TTS for free with no setup, so effort went into the part that matters: the agent logic, tool calling and
guardrails. FastAPI serves the API and the frontend from one service, and Groq's fast inference keeps latency low.

**2. Most difficult part?**
`<fill in from your own experience, e.g. echo and turn-taking between mic and TTS, or keeping the LLM inside policy>`

**3. One more week?**
Server-side streaming STT/TTS (for better accents, a consistent voice and lower latency), barge-in, and
Hinglish support, plus an evaluation set of policy test conversations to catch prompt regressions.

**4. 1,000 conversations a day?**
Persistent session storage (Redis) instead of in-memory, a real order system behind the tools, authentication
of callers before sharing order data, rate limiting and retries across LLM providers, logging and monitoring
of latency, tool errors and policy violations, automated evaluation, escalation to a human agent, and a
paid STT/TTS and LLM tier with cost tracking.
