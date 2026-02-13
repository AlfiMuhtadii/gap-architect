# Career Gap Analyzer - Production-Ready MVP

## Architecture Overview

This system demonstrates production-ready engineering practices
for an AI-powered application within MVP constraints.

### Core Design Principles

**1. Reliability Over Features**
- Multi-layer LLM fallback (cloud → local → heuristic)
- Graceful degradation under failure
- Always returns actionable results

**2. Async-First Architecture**
- Non-blocking API endpoints (immediate 201 response)
- Background processing for LLM calls
- Client-side polling for status updates

**3. Validation at Every Layer**
- Input validation (Pydantic)
- Output validation (schema + content)
- Automatic repair for malformed AI responses
- Graceful failure states

**4. Performance Optimization**
- Deterministic caching (fingerprint-based)
- Zero redundant LLM calls for identical inputs
- Efficient database queries with proper indexes

**5. Observability**
- Comprehensive logging (llm_runs audit trail)
- Status tracking throughout pipeline
- Error context for debugging

---

## Engineering Decisions

### Why Multi-Layer LLM Fallback?

**Real Production Issues:**
- API rate limits (50-3500 req/min depending on tier)
- Service downtime (~5-10 incidents/year per provider)
- Timeout failures during peak load
- Cost optimization needs at scale

**Implementation:**
- Lightweight local model (Llama 3B via Ollama)

### Why Async Processing?

**Per assignment requirement:**
> "The API must return immediately; AI processing happens in background"

**Implementation:**
- FastAPI BackgroundTasks (built-in)
- PostgreSQL for state management
- No complex queue infrastructure needed for MVP

**Benefit:**
User never waits on slow LLM calls (better UX).

### Why This Validation Strategy?

**AI outputs are unreliable:**
- ~1-5% malformed JSON
- Occasional hallucinated skills
- Missing required fields

**Our approach:**
1. Strict Pydantic validation
2. Attempt automatic repair
3. Verify skills against source JD
4. Fail gracefully with clear errors

**Result:**
Robust system that handles AI unreliability without crashing.
